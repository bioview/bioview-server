# Known BIOPAC USB Vendor IDs and Product IDs
BIOPAC_VENDOR_ID = 0x097E

BIOPAC_CONNECTION_CODES = {
    1: "MPSUCCESS",
    2: "MPDRVERR",
    3: "MPDLLBUSY",
    4: "MPINVPARA",
    5: "MPNOTCON",
    6: "MPREADY",
    7: "MPWPRETRIG",
    8: "MPWTRIG",
    9: "MPBUSY",
    10: "MPNOACTCH",
    11: "MPCOMERR",
    12: "MPINVTYPE",
    13: "MPNOTINNET",
    14: "MPSMPLDLERR",
    15: "MPMEMALLOCERR",
    16: "MPSOCKERR",
    17: "MPUNDRFLOW",
    18: "MPPRESETERR",
    19: "MPPARSEERR",
}


#: What an mpdev result code means in practice, for the ones a user can act on.
#: A bare "Error Code: 2" in the log says nothing; the cause and the fix do.
BIOPAC_CODE_EXPLANATIONS = {
    "MPDRVERR": (
        "the MP device driver did not respond -- the unit is plugged in but "
        "Windows cannot drive it. Check Device Manager for a warning on the "
        "BIOPAC unit and reinstall its driver"
    ),
    "MPDLLBUSY": (
        "another program is already connected to the MP unit; close it and retry"
    ),
    "MPNOTCON": "no MP unit is connected",
    "MPINVPARA": "the MP unit rejected a parameter (model, connection type or port)",
    "MPNOACTCH": "no acquisition channels are enabled for this device",
    "MPCOMERR": "communication with the MP unit failed",
    "MPINVTYPE": "the configured MP model does not match the attached unit",
    "MPNOTINNET": "the MP unit was not found on the network",
    "MPSOCKERR": "a network socket error occurred talking to the MP unit",
}


def describe_biopac_code(result_code) -> str:
    """A readable description of an mpdev result code.

    Renders as e.g. "MPDRVERR (code 2): the MP device driver did not respond..."
    so a failure explains itself instead of surfacing a bare number.
    """
    name = BIOPAC_CONNECTION_CODES.get(result_code)
    if name is None:
        return f"unknown error (code {result_code})"

    explanation = BIOPAC_CODE_EXPLANATIONS.get(name)
    if explanation is None:
        return f"{name} (code {result_code})"
    return f"{name} (code {result_code}): {explanation}"
