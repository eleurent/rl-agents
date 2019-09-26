import collections
from gym.core import Env


class Configurable(object):
    """
        This class is a container for a configuration dictionary.
        It allows to provide a default_config function with pre-filled configuration.
        When provided with an input configuration, the default one will recursively be updated,
        and the input configuration will also be updated with the resulting configuration.
    """
    def __init__(self, config=None):
        self.config = self.default_config()
        if config:
            # Override default config with variant
            Configurable.rec_update(self.config, config)
            # Override incomplete variant with completed variant
            Configurable.rec_update(config, self.config)

    @classmethod
    def default_config(cls):
        """
            Override this function to provide the default configuration of the child class
        :return: a configuration dictionary
        """
        return {}

    @staticmethod
    def rec_update(d, u):
        """
            Recursive update of a mapping
        :param d: a mapping
        :param u: a mapping
        :return: d updated recursively with u
        """
        for k, v in u.items():
            if isinstance(v, collections.Mapping):
                d[k] = Configurable.rec_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d


class Dummy(object):
    pass


_ignored_keys = set(Dummy.__dict__.keys())


class Serializable(dict):
    """
        Automatically serialize all fields of an object to a dictionary.

    Keys correspond to field names, and values correspond to field values representation by default but are
    recursively expanded to sub-dictionaries for any Serializable field.
    """
    def to_dict(self):
        d = dict()
        for (key, value) in self.__dict__.items():
            if key not in _ignored_keys:
                if isinstance(value, Serializable):
                    d[key] = value.to_dict()
                else:
                    d[key] = repr(value)
        return d

    def from_dict(self, dictionary):
        for (key, value) in dictionary.items():
            if key in self.__dict__:
                if isinstance(value, Serializable):
                    self.__dict__[key].from_config(dictionary)
                else:
                    self.__dict__[key] = value


def serialize(obj):
    """
        Serialize any object to a dictionary, so that it can be dumped easily to a JSON file.

     Four rules are applied:
        - To be able to recreate the object, specify its class, or its spec id if the object is an Env.
        - If the object has a config dictionary field, use it. It is assumed that this config suffices to recreate a
        similar object.
        - If the object is Serializable, use its recursive conversion to a dictionary.
        - Else, use its __dict__ by applying repr() on its values
    :param obj: an object
    :return: a dictionary describing the object
    """
    if hasattr(obj, "config"):
        d = obj.config
    elif isinstance(obj, Serializable):
        d = obj.to_dict()
    else:
        d = {key: repr(value) for (key, value) in obj.__dict__.items()}
    d['__class__'] = repr(obj.__class__)
    if isinstance(obj, Env):
        d['id'] = obj.spec.id
        d['import_module'] = getattr(obj, "import_module", None)
    return d

