class JointError(Exception):
    """Base class for all public JOINT errors."""


class ConfigurationError(JointError):
    pass


class InputFormatError(JointError):
    pass


class PeakAlignmentError(JointError):
    pass


class SegmentationError(JointError):
    pass


class RegistrationError(JointError):
    pass


class QuantificationError(JointError):
    pass


class OptionalDependencyError(JointError):
    pass
