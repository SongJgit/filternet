from mmengine.registry import Registry

PARAMS = Registry('params', scope='filternet', locations=['filternet.params'])
MODELS = Registry('models', scope='filternet', locations=['filternet.models'])
DATASETS = Registry('datasets',
                    scope='filternet',
                    locations=['filternet.datasets'])
MODELTRANS = Registry('model_trans',
                      scope='filternet',
                      locations=['filternet.models.base'])
TRANSFORMS = Registry('transforms',
                      scope='filternet',
                      locations=['filternet.datasets.transforms'])
PARAM_SCHEDULERS = Registry('param_scheduler',
                            scope='filternet',
                            locations=['filternet.param_schedulers'])
FILTER = Registry('filter',
                  scope='filternet',
                  locations=['filternet.models.filter'])
DYNAMICSMODEL = Registry('dynamics_model',
                         scope='filternet',
                         locations=['filternet.models.filter'])
MEASUREMENTMODEL = Registry('measurement_model',
                            scope='filternet',
                            locations=['filternet.models.filter'])

OPTIMIZER = Registry('optimizer',
                     scope='filternet',
                     locations=['filternet.models.optimizer'])
