class EarlyStopper:
    def __init__(self, train_direction: str = "decrease"):
        if train_direction == "decrease":
            self.prev_value = float("inf")
            self.this_value = float("inf")
        elif train_direction == "increase":
            self.prev_value = float("-inf")
            self.this_value = float("-inf")
        self.train_direction = train_direction

    def check(self, to_track: float):
        self.prev_value = self.this_value
        self.this_value = to_track
        continue_train = self._compare()

        return continue_train

    def _compare(self):
        continue_train = False
        if self.train_direction == "decrease":
            if self.this_value <= self.prev_value:
                continue_train = True
        elif self.train_direction == "increase":
            if self.this_value >= self.prev_value:
                continue_train = True

        return continue_train