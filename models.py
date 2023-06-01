from simulator import Kernel
from handlers import start_simulation


class Model:
    reader = None
    tags = None
    transaction = None
    max_tags_num = None
    protocol = None

    def __init__(self):
        self.tags = []
        self.num_tags_simulated = 0
        self.protocol = None

def simulate_tags(encoding, tari, trext, q, num_tags, ber, protocol):
    match protocol:
        case 'PEFSA':
            from objects.pefsa import Reader
            from objects.pefsa import Generator
        case 'CHEN':
            from objects.chen import Reader
            from objects.chen import Generator
        case 'ILCM':
            from objects.ilcm import Reader
            from objects.ilcm import Generator
        case 'ADAPTIVE':
            from objects.adaptive import Reader
            from objects.adaptive import Generator
        case 'FAST':
            from objects.fast import Reader
            from objects.fast import Generator
        case 'SUBEP':
            from objects.subep import Reader
            from objects.subep import Generator
    
    
    # 0) Building the model
    model = Model()
    model.max_tags_num = num_tags

    # 1) Building the reader
    reader = Reader()
    reader.q = q
    reader.trext = trext
    reader.tari = tari
    reader.tag_encoding = encoding
    reader.ber = ber
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