import abc
from custodian.custodian import ErrorHandler, Validator


#TODO: do we stick to custodian's ErrorHandler/Validator inheritance ??

class SRCErrorHandler(ErrorHandler):

    HANDLER_PRIORITIES = {'PRIORITY_FIRST': 0,
                          'PRIORITY_VERY_HIGH': 1,
                          'PRIORITY_HIGH': 2,
                          'PRIORITY_MEDIUM': 3,
                          'PRIORITY_LOW': 4,
                          'PRIORITY_VERY_LOW': 5,
                          'PRIORITY_LAST': 6}
    PRIORITY_FIRST = HANDLER_PRIORITIES['PRIORITY_FIRST']
    PRIORITY_VERY_HIGH = HANDLER_PRIORITIES['PRIORITY_VERY_HIGH']
    PRIORITY_HIGH = HANDLER_PRIORITIES['PRIORITY_HIGH']
    PRIORITY_MEDIUM = HANDLER_PRIORITIES['PRIORITY_MEDIUM']
    PRIORITY_LOW = HANDLER_PRIORITIES['PRIORITY_LOW']
    PRIORITY_VERY_LOW = HANDLER_PRIORITIES['PRIORITY_VERY_LOW']
    PRIORITY_LAST = HANDLER_PRIORITIES['PRIORITY_LAST']


    def __init__(self):
        self.fw_spec = None
        self.fw_to_check = None

    @abc.abstractmethod
    def as_dict(self):
        pass

    @abc.abstractmethod
    def from_dict(cls, d):
        pass

    @abc.abstractmethod
    def setup(self):
        pass

    def set_fw_spec(self, fw_spec):
        self.fw_spec = fw_spec

    def set_fw_to_check(self, fw_to_check):
        self.fw_to_check = fw_to_check

    def src_setup(self, fw_spec, fw_to_check):
        self.set_fw_spec(fw_spec=fw_spec)
        self.set_fw_to_check(fw_to_check=fw_to_check)
        self.setup()

    @abc.abstractproperty
    def handler_priority(self):
        pass

    @property
    def skip_remaining_handlers(self):
        return False

    @abc.abstractproperty
    def allow_fizzled(self):
        pass

    @abc.abstractproperty
    def allow_completed(self):
        pass

    @abc.abstractmethod
    def has_corrections(self):
        pass


class MonitoringSRCErrorHandler(ErrorHandler):

    HANDLER_PRIORITIES = {'PRIORITY_FIRST': 0,
                          'PRIORITY_VERY_HIGH': 1,
                          'PRIORITY_HIGH': 2,
                          'PRIORITY_MEDIUM': 3,
                          'PRIORITY_LOW': 4,
                          'PRIORITY_VERY_LOW': 5,
                          'PRIORITY_LAST': 6}
    PRIORITY_FIRST = HANDLER_PRIORITIES['PRIORITY_FIRST']
    PRIORITY_VERY_HIGH = HANDLER_PRIORITIES['PRIORITY_VERY_HIGH']
    PRIORITY_HIGH = HANDLER_PRIORITIES['PRIORITY_HIGH']
    PRIORITY_MEDIUM = HANDLER_PRIORITIES['PRIORITY_MEDIUM']
    PRIORITY_LOW = HANDLER_PRIORITIES['PRIORITY_LOW']
    PRIORITY_VERY_LOW = HANDLER_PRIORITIES['PRIORITY_VERY_LOW']
    PRIORITY_LAST = HANDLER_PRIORITIES['PRIORITY_LAST']

    @abc.abstractmethod
    def as_dict(self):
        pass

    @abc.abstractmethod
    def from_dict(cls, d):
        pass

    @abc.abstractproperty
    def handler_priority(self):
        pass

    @property
    def skip_remaining_handlers(self):
        return False


class SRCValidator(Validator):

    HANDLER_PRIORITIES = {'PRIORITY_FIRST': 0,
                          'PRIORITY_VERY_HIGH': 1,
                          'PRIORITY_HIGH': 2,
                          'PRIORITY_MEDIUM': 3,
                          'PRIORITY_LOW': 4,
                          'PRIORITY_VERY_LOW': 5,
                          'PRIORITY_LAST': 6}
    PRIORITY_FIRST = HANDLER_PRIORITIES['PRIORITY_FIRST']
    PRIORITY_VERY_HIGH = HANDLER_PRIORITIES['PRIORITY_VERY_HIGH']
    PRIORITY_HIGH = HANDLER_PRIORITIES['PRIORITY_HIGH']
    PRIORITY_MEDIUM = HANDLER_PRIORITIES['PRIORITY_MEDIUM']
    PRIORITY_LOW = HANDLER_PRIORITIES['PRIORITY_LOW']
    PRIORITY_VERY_LOW = HANDLER_PRIORITIES['PRIORITY_VERY_LOW']
    PRIORITY_LAST = HANDLER_PRIORITIES['PRIORITY_LAST']
    pass