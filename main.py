import click
import epcstd as std
import models as models
from dataclasses import dataclass
from handlers import set_transaction_file

DEFAULT_ENCODING = 'M2'        # FM0, M2, M4, M8
DEFAULT_TARI = "6.25"          # 6.25, 12.5, 18.75, 25
DEFAULT_USE_TREXT = False      # don't use extended preamble
DEFAULT_Q = 4                  # Q parameter
DEFAULT_NUM_TAGS = 1000       # number of tags to simulate
DEFAULT_BER = 0.0
DEFAULT_PROTOCOL = "ADAPTIVE"

@dataclass
class ModelInput:
    encoding: str = DEFAULT_ENCODING
    tari: str = DEFAULT_TARI
    use_trext: bool = DEFAULT_USE_TREXT
    q: int = DEFAULT_Q
    num_tags: int = DEFAULT_NUM_TAGS
    ber: float = DEFAULT_BER
    protocol: str = DEFAULT_PROTOCOL

@click.command()

@click.option(
    "-m", "--encoding", type=click.Choice(["1", "2", "4", "8"]),
    default=DEFAULT_ENCODING, help="Tag encoding", show_default=True
)
@click.option(
    "-t", "--tari", default=DEFAULT_TARI, show_default=True,
    type=click.Choice(["6.25", "12.5", "18.75", "25"]), help="Tari value"
)
@click.option(
    "--trext/--no-trext", "use_trext", default=DEFAULT_USE_TREXT,
    show_default=True, help="Use extended preamble in tag response"
)
@click.option(
    "-q", default=DEFAULT_Q, show_default=True, help="Q parameter"
)
@click.option(
    "-n", "--num-tags", default=DEFAULT_NUM_TAGS, show_default=True, help="Number of tags to simulate"
)
@click.option(
    "-b", "--ber", default=DEFAULT_BER, show_default=True, help="Bit error rate"
)
@click.option(
    "-p", "--protocol", default=DEFAULT_PROTOCOL, show_default=True, 
    type=click.Choice(["ADAPTIVE", "FAST", "CHEN", "ILCM", "SUBEP", "PEFSA"]), help="Protocol to analyse"
)
def start_single(**kwargs):
    model_input = ModelInput(**kwargs)
    set_transaction_file(model_input.protocol)
    ret = estimate_rates(
        encoding=model_input.encoding,
        tari=float(model_input.tari) * 1e-6,
        trext=model_input.use_trext,
        q=model_input.q,
        num_tags = model_input.num_tags,
        ber = model_input.ber,
        protocol = model_input.protocol,
    )

def estimate_rates(encoding , tari , trext, q, num_tags, ber, protocol):
    # If encoding is given as a string, try to parse it (otherwise assume
    # it is given as a TagEncoding value)
    print("[+] tari={}, m={}, "
          "trext={}, q={}, tags_amount={}, ber={}, protocol={}"
          "".format(tari, str(encoding), 
                    trext,  q, num_tags, ber, protocol ))

    try:
        encoding = parse_tag_encoding(encoding)
    except ValueError:
        pass
    result = models.simulate_tags(
        encoding = encoding, tari = tari, trext = trext, 
        q = q, num_tags = num_tags, ber = ber, protocol = protocol
    )

    # result['m'] = encoding.name
    # result['tari'] = tari
    # result['speed'] = speed
    # result['orientation'] = orientation
    # result['doppler'] = doppler
    # result['trext'] = trext
    # result['angle'] = angle
    # result['q'] = q
    # result['frequency'] = frequency
    # return result

def parse_tag_encoding(s):
    s = s.upper()
    if s in {'1', "FM0"}:
        return std.TagEncoding.FM0
    elif s in {'2', 'M2'}:
        return std.TagEncoding.M2
    elif s in {'4', 'M4'}:
        return std.TagEncoding.M4
    elif s in {'8', 'M8'}:
        return std.TagEncoding.M8
    else:
        raise ValueError('illegal encoding = {}'.format(s))

if __name__ == '__main__':
    start_single()
