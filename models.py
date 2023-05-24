from objects import Reader
from objects import Generator
from simulator import Kernel
from handlers import start_simulation


class Model:
    reader = None
    tags = None
    transaction = None
    max_tags_num = None

    def __init__(self):
        self.reader = Reader()
        self.tags = []
        self.transaction = None
        self.num_tags_simulated = 0

def simulate_tags():

    # 0) Building the model
    model = Model()
    model.max_tags_num = 5000

    # 1) Building the reader
    reader = Reader()
    model.reader = reader

    # 2) Building tags
    tag_generator = Generator()
    tags = []

    for i in range(0, model.max_tags_num):
        tag = tag_generator.create_tag(i)
        tag.setup()
        tag._power_on()
        tags.append(tag)
    model.tags = tags

    # 3) Launching simulation
    kernel = Kernel()
    kernel.context = model
    kernel.run(start_simulation)

simulate_tags()