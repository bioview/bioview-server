# TODO: Move to client. Instructions are purely UI 

import time
from pathlib import Path

import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

from PyQt6.QtCore import QObject, QThread, pyqtSignal # TODO: Remove. No Qt components in server

from bioview_server import ExperimentConfiguration


# Each instruction handler should be its own QObject
class AudioPlayer(QObject):
    def __init__(self, config: ExperimentConfiguration, parent=None):
        super().__init__(parent=parent)
        self.instruction_file = config.get_param("instruction_file", None)
        if self.instruction_file is None or not Path(self.instruction_file).exists():
            raise Exception("No valid audio file found.")

        self.loop_instruction = config.get_param("loop_instructions", True)
        self._should_stop = False

        # Initialize pygame mixer
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init()
        except pygame.error as e:
            print(f"Warning: pygame mixer init issue: {e}")

    def pre_run(self):
        self._should_stop = False  # Reset stop flag
        try:
            pygame.mixer.music.load(self.instruction_file)
        except Exception as e:
            print(f"Error loading audio file: {e}")

    def run(self):
        if self._should_stop:
            return True
        try:
            pygame.mixer.music.play()
        except Exception as e:
            print(f"Error playing audio: {e}")
            return True

        # Check for completion or stop signal without holding the mutex
        while True:
            if self._should_stop:
                pygame.mixer.music.stop()
                return True
            if not pygame.mixer.music.get_busy():
                # Audio finished naturally
                break

            # Sleep briefly before checking again
            self.thread().msleep(100)

        # Audio completed - check if we should loop
        return not self.loop_instruction  # True if should stop

    def stop(self):
        self._should_stop = True
        try:
            pygame.mixer.music.stop()
        except Exception:
            pass


class TextInstructions(QObject):
    textUpdate = pyqtSignal(str)

    def __init__(self, config: ExperimentConfiguration, parent=None):
        super().__init__(parent)
        f_path = Path(config.get_param("instruction_file", ""))

        try:
            self.instructions = open(f_path).read().splitlines()
        except Exception:
            return

        self.current_index = 0
        self.loop_instruction = config.get_param("loop_instructions", True)
        self.interval = config.get_param("instruction_interval", 5)  # seconds
        self._should_stop = False

    def pre_run(self):
        self._should_stop = False

    def run(self):
        # Check if we should stop
        if self._should_stop:
            return True

        # Check if we've reached the end
        if self.current_index >= len(self.instructions):
            if self.loop_instruction:
                self.current_index = 0
            else:
                return True  # Should stop

        # Send signal to update text
        instruction_text = self.instructions[self.current_index]
        self.current_index += 1
        self.textUpdate.emit(instruction_text)

        # Wait for the specified interval, but check for stop periodically
        elapsed = 0
        sleep_increment = 0.1  # Check every 100ms
        while elapsed < self.interval and not self._should_stop:
            time.sleep(sleep_increment)
            elapsed += sleep_increment

        return self._should_stop  # Return True if we should stop

    def stop(self):
        self._should_stop = True


class InstructionWorker(QThread):
    logEvent = pyqtSignal(str, str)
    textUpdate = pyqtSignal(str)  # Signal to update instruction text
    showDialog = pyqtSignal()  # Signal to show the dialog
    hideDialog = pyqtSignal()  # Signal to hide the dialog

    def __init__(
        self, config: ExperimentConfiguration, running: bool = False, parent=None
    ):
        super().__init__(parent)
        self.config = config

        self.instruction_type = config.get_param("instruction_type")
        self.instruction_handler = None
        self.running = running

        try:
            if self.instruction_type == "text":
                self.instruction_handler = TextInstructions(config, parent=self)
                self.instruction_handler.textUpdate.connect(self.textUpdate)
            elif self.instruction_type == "audio":
                self.instruction_handler = AudioPlayer(config, parent=self)
            else:
                raise Exception(f"Invalid instruction type: {self.instruction_type}")
        except Exception as e:
            self.logEvent.emit(
                "error", f"Unable to initialize instruction handler: {e}"
            )

    def run(self):
        if self.instruction_handler is None:
            self.logEvent.emit("warning", "No instruction handler available")
            return

        self.running = True

        # This may include file-reading, etc one time tasks before running that won't loop
        if hasattr(self.instruction_handler, "pre_run"):
            self.instruction_handler.pre_run()

        if self.instruction_type == "text":
            self.showDialog.emit()

        while self.running:
            should_stop = self.instruction_handler.run()
            if should_stop:
                break

        self.stop()

    def stop(self):
        self.running = False

        if self.instruction_handler is not None:
            self.instruction_handler.stop()

        # Hide text dialog
        if self.instruction_type == "text":
            self.hideDialog.emit()
