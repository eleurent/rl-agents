
class Config(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


class Dummy(object):
    pass


_ignored_keys = set(Dummy.__dict__.keys())


class Configurable(dict):
    def to_dict(self):
        d = dict()
        for (key, value) in self.__dict__.items():
            if key not in _ignored_keys:
                if isinstance(value, Configurable):
                    d[key] = value.to_dict()
                else:
                    d[key] = repr(value)
        return d

    def from_dict(self, dictionary):
        for (key, value) in dictionary.items():
            if key in self.__dict__:
                if isinstance(value, Configurable):
                    self.__dict__[key].from_dict(dictionary)
                else:
                    self.__dict__[key] = value


def serialize(obj):
    if isinstance(obj, Configurable):
        d = obj.to_dict()
    else:
        d = {key: repr(value) for (key, value) in obj.__dict__.items()}
    d['__class__'] = repr(obj.__class__)
    return d

