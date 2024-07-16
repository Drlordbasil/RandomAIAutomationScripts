import random


class Plant:
    def __init__(self):
        self.position = (random.randint(0, 99), random.randint(0, 99))
        self.growth_stage = 0
        self.growth_rate = random.uniform(0.1, 0.5)

    def grow(self):
        if self.growth_stage < 5:
            self.growth_stage += self.growth_rate
