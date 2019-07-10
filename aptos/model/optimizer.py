from torch import optim


class MultiOptimizer:
    """
    Class to wrap optimizers for both the model and the loss function.
    """
    def __init__(self, model_params, loss_params, model_config, loss_config):
        # setup model optimizer
        self._model_opt = self._build(model_params, model_config)

        # setup loss optimizer
        self._loss_opt = self._build(loss_params, loss_config)

    def _build(self, params, config):
        return getattr(optim, config['type'])(params, **config['args'])

    @property
    def model_opt(self):
        return self._model_opt

    @property
    def loss_opt(self):
        return self._loss_opt

    @property
    def model_lr(self):
        for param_group in self.model_opt.param_groups:
            return param_group['lr']

    @property
    def loss_lr(self):
        for param_group in self.loss_opt.param_groups:
            return param_group['lr']

    @property
    def models(self):
        return {'model_opt': self.model_opt, 'loss_opt': self.loss_opt}

    def state_dict(self):
        return {name: model.state_dict() for name, model in self.models.items()}

    def load_state_dict(self, checkpoint):
        for name, model in self.models.items():
            model.load_state_dict(checkpoint[name])

    def zero_grad(self):
        for name, model in self.models.items():
            model.zero_grad()

    def step(self):
        for name, model in self.models.items():
            model.step()
