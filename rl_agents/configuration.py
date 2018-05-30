import collections
from gym.core import Env
from gym.envs.registration import registry

class Configurable(object):
    """
        This class is a container for a configuration dictionary.
        It allows to provide a default_config function with prefilled configuration.
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
                    self.__dict__[key].from_dict(dictionary)
                else:
                    self.__dict__[key] = value


def serialize(obj):
    if hasattr(obj, "config"):
        d = obj.config
    elif isinstance(obj, Serializable):
        d = obj.to_dict()
    else:
        d = {key: repr(value) for (key, value) in obj.__dict__.items()}
    d['__class__'] = repr(obj.__class__)
    if isinstance(obj, Env):
        env_index = list(registry.all()).index(obj.spec)
        env_id = list(registry.env_specs.keys())[env_index]
        d['id'] = env_id
    return d

