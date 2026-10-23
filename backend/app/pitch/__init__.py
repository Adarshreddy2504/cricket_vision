"""
CricketVision — pitch detection and calibration.

The public API of the pitch module:

    app.pitch.errors        the failure vocabulary
    app.pitch.constants     every tunable the module applies
    app.pitch.geometry      the pitch's dimensions and their justification
    app.pitch.corners       the four corners, their canonical order, their checks
    app.pitch.calibration   the calibration record and its status
    app.pitch.homography    the pixel -> pitch conversion
    app.pitch.config        where the detector's weights live
    app.pitch.mask_geometry a mask boundary -> four corners
    app.pitch.detector      the detector interface and its YOLO implementations
    app.pitch.adapter       projecting a Part 1 trajectory into pitch space

Import from the submodules directly. This package deliberately re-exports
nothing: the names above are the module's surface, and each one says what it
is for.
"""
