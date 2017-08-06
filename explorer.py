class LinearDecayExplorer:
    def __init__(self, final_exploration_step=10**6,
                start_epsilon=1.0, final_epsilon=0.1):
        self.final_exploration_step = final_exploration_step
        self.start_epsilon = start_epsilon
        self.final_epsilon = final_epsilon
        self.base_epsilon = self.start_epsilon - self.final_epsilon

    def value(self, t):
        if t > self.final_exploration_step:
            return self.final_epsilon
        factor = 1 - float(t) / self.final_exploration_step
        return self.base_epsilon * factor + self.final_epsilon
