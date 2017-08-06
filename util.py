import tensorflow as tf
import collections

ALREADY_INITIALIZED = set()

def initialize():
    new_variables = set(tf.global_variables()) - ALREADY_INITIALIZED
    get_session().run(tf.variables_initializer(new_variables))
    ALREADY_INITIALIZED.update(new_variables)


def scope_vars(scope, trainable_only=False):
    return tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES if trainable_only else tf.GraphKeys.GLOBAL_VARIABLES,
        scope=scope if isinstance(scope, str) else scope.name
    )

def scope_name():
    return tf.get_variable_scope().name

def absolute_scope_name(relative_scope_name):
    return scope_name() + '/' + relative_scope_name

def huber_loss(x, delta=1.0):
    return tf.where(
        tf.abs(x) < delta,
        tf.square(x) * 0.5,
        delta * (tf.abs(x) - 0.5 * delta)
    )

def get_session():
    return tf.get_default_session()

def function(inputs, outputs, updates=None, givens=None):
    if isinstance(outputs, list):
        return _Function(inputs, outputs, updates, givens=givens)
    elif isinstance(outputs, (dict, collections.OrderedDict)):
        f = _Function(inputs, outputs.values(), updates, givens=givens)
        return lambda *args, **kwargs: type(outputs)(zip(outputs.keys(), f(*args, **kwargs)))
    else:
        f = _Function(inputs, [outputs], updates, givens=givens)
        return lambda *args, **kwargs: f(*args, **kwargs)[0]

class _Function(object):
    def __init__(self, inputs, outputs, updates, givens):
        self.inputs = inputs
        self.outputs = outputs
        self.givens = {} if givens is None else givens
        updates = updates or []
        self.update_group = tf.group(*updates)
        self.outputs_update = list(outputs) + [self.update_group]

    def __call__(self, *args, **kwargs):
        feed_dict = {}
        for inpt, value in zip(self.inputs, args):
            feed_dict[inpt] = value
        kwargs_passed_inpt_names = set()
        for inpt in self.inputs[len(args):]:
            inpt_name = inpt.name.split(':')[0]
            inpt_name = inpt_name.split('/')[-1]
            if inpt_name in kwargs:
                kwargs_passed_inpt_names.add(inpt_name)
                feed_dict[inpt] = kwargs.pop(inpt_name)
        for inpt in self.givens:
            feed_dict[inpt] = feed_dict.get(inpt, self.givens[inpt])
        results = get_session().run(self.outputs_update, feed_dict=feed_dict)[:-1]
        return results
