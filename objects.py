import enum
import itertools
import numpy as np
import epcstd as std
from simulator import Kernel

# ===========================================================================
# Reader States
# ===========================================================================

class _ReaderState:
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name

    def __str__(self):
        return self._name
    
    def get_timeout(self, reader):
        raise NotImplementedError

    def enter(self, reader):
        raise NotImplementedError
    
    def handle_turn_on(self, reader):
        raise NotImplementedError

    def handle_turn_off(self, reader):
        raise NotImplementedError

    def handle_timeout(self, reader):
        raise NotImplementedError

    def handle_query_reply(self, reader, frame):
        raise NotImplementedError
    
    def handle_ack_reply(self, reader, frame):
        raise NotImplementedError
    
    def handle_query_adjust(self, reader):
        raise NotImplementedError


class _ReaderOFF(_ReaderState):
    def __init__(self):
        super().__init__('OFF')

    def get_timeout(self, reader):
        return 0

    def enter(self, reader):
        reader.last_rn = None
        reader.stop_round()
        return None

    def handle_turn_on(self, reader):
        slot = reader.next_slot()
        return reader.set_state(slot.first_state)

    def handle_turn_off(self, reader):
        return None  # Already off

    def handle_timeout(self, reader):
        return reader.turn_on()  # The default action on timeout is turn on

    def handle_query_reply(self, reader, frame):
        return None  # Any reply ignored
    
    def handle_ack_reply(self, reader, frame):
        return None  # Any reply ignored
    
    def handle_query_adjust(self, reader):
        return None  # Can not send commands when off


class _ReaderQuery(_ReaderState):
    def __init__(self):
        super().__init__('QUERY')

    def get_timeout(self, reader):
        t_cmd = std.query_duration(reader.tari, reader.rtcal, reader.trcal,
                                   reader.delim, reader.dr, reader.tag_encoding,
                                   reader.trext, reader.sel, reader.session,
                                   reader.target, reader.q)
        t1 = std.link_t1_max(reader.rtcal, reader.trcal, reader.dr, reader.temp)
        t3 = std.link_t3()
        return t_cmd + t1 + t3

    def enter(self, reader):
        reader.empty_slots = 0
        reader.collision_slots = 0
        reader.single_slots = 0
        reader.last_rn = None
        cmd = std.Query(reader.dr, reader.tag_encoding, reader.trext,
                        reader.sel, reader.session, reader.target, reader.q)
        return std.ReaderFrame(reader.preamble, cmd)
    
    def handle_turn_on(self, reader):
        return None  # Reader is already ON

    def handle_turn_off(self, reader):
        # All actions are performed in OFF.enter(), just move there
        return reader.set_state(Reader.State.OFF)
    
    def handle_timeout(self, reader):
        slot = reader.next_slot()
        return reader.set_state(slot.first_state)
    
    def handle_query_reply(self, reader, frame):
        reader.last_rn = frame.reply.rn
        return reader.set_state(Reader.State.ACK)
    
    def handle_ack_reply(self, reader, frame):
        raise RuntimeError("unexpected AckReply in QUERY state")
    
    def handle_query_adjust(self, reader):
        raise RuntimeError("unexpected QADJUST in QUERY state")

    
class _ReaderQREP(_ReaderState):
    def __init__(self): 
        super().__init__('QREP')

    def get_timeout(self, reader):
        t_cmd = std.query_rep_duration(reader.tari, reader.rtcal, reader.trcal,
                                       reader.delim, reader.session)
        t1 = std.link_t1_max(reader.rtcal, reader.trcal, reader.dr,
                             reader.temp)
        t3 = std.link_t3()
        return t_cmd + t1 + t3

    def enter(self, reader):
        reader.last_rn = None
        cmd = std.QueryRep(reader.session)
        return std.ReaderFrame(reader.sync, cmd)
    
    def handle_turn_on(self, reader):
        return None  # Reader is already ON

    def handle_turn_off(self, reader):
        # All actions are performed in OFF.enter(), just move there
        return reader.set_state(Reader.State.OFF)

    def handle_timeout(self, reader):
        slot = reader.next_slot()
        return reader.set_state(slot.first_state)

    def handle_query_reply(self, reader, frame):
        reader.last_rn = frame.reply.rn
        return reader.set_state(Reader.State.ACK)
    
    def handle_ack_reply(self, reader, frame):
        raise RuntimeError("unexpected AckReply in QREP state")
    
    def handle_query_adjust(self, reader):
        raise RuntimeError("unexpected QADJUST in QREP state")
    

class _ReaderQADJUST(_ReaderState):
    def __init__(self): super().__init__('QADJUST')

    def get_timeout(self, reader):
        t_cmd = std.query_adjust_duration(reader.tari, reader.rtcal, reader.trcal,
                                       reader.delim, reader.session, reader.upDn)
        t1 = std.link_t1_max(reader.rtcal, reader.trcal, reader.dr,
                             reader.temp)
        t3 = std.link_t3()
        return t_cmd + t1 + t3

    def enter(self, reader):
        cmd = std.QueryAdjust(reader.session, reader.upDn)
        return std.ReaderFrame(reader.preamble, cmd)

    def handle_turn_on(self, reader): return None

    def handle_turn_off(self, reader):
        return reader.set_state(Reader.State.OFF)

    def handle_timeout(self, reader):
        slot = reader.next_slot()
        return reader.set_state(slot.first_state)
    
    def handle_query_adjust(self, reader):
        reader.q = reader.q + reader.upDn.eval()
        #reader.set_state(Reader.State.QREP)

    def handle_query_reply(self, reader, frame):
        reader.last_rn = frame.reply.rn
        return reader.set_state(Reader.State.ACK)

    def handle_ack_reply(self, reader, frame):
        raise RuntimeError("unexpected AckReply in QADJUST state")
    
    def handle_query_adjust(self, reader):
        reader.q = reader.q + reader.upDn.eval()
        #reader.set_state(Reader.State.QREP)


class _ReaderACK(_ReaderState):
    def __init__(self): super().__init__('ACK')

    def get_timeout(self, reader):
        t_cmd = std.ack_duration(reader.tari, reader.rtcal, reader.trcal,
                                 reader.delim, reader.last_rn)
        t1 = std.link_t1_max(reader.rtcal, reader.trcal, reader.dr,
                             reader.temp)
        t3 = std.link_t3()
        return t_cmd + t1 + t3

    def enter(self, reader):
        cmd = std.Ack(reader.last_rn)
        return std.ReaderFrame(reader.sync, cmd)

    def handle_turn_on(self, reader): return None

    def handle_turn_off(self, reader):
        return reader.set_state(Reader.State.OFF)

    def handle_timeout(self, reader):
        slot = reader.next_slot()
        return reader.set_state(slot.first_state)

    def handle_query_reply(self, reader, frame):
        raise RuntimeError("unexpected RN16 in ACK state")

    def handle_ack_reply(self, reader, frame):
        reader.epc_bank.append(frame.reply.epc)
        if reader.read_tid_bank:
            return reader.set_state(Reader.State.REQRN)
        else:
            slot = reader.next_slot()
            return reader.set_state(slot.first_state)

    def handle_query_adjust(self, reader):
        raise RuntimeError("unexpected QADJUST in ACK state")


# ===========================================================================
# Rounds and slots
# ===========================================================================
class _ReaderSlot:
    def __init__(self, owner, index, first_state):
        self._owner, self._index, self._first_state = owner, index, first_state

    @property
    def first_state(self):
        return self._first_state

    @property
    def index(self):
        return self._index

    @property
    def owner(self):
        return self._owner


class _ReaderRound:
    def __init__(self, reader, index):
        self._index = index

        def slots_gen():
            # yield _ReaderSlot(self, 0, Reader.State.QUERY)
            # for i in range(1, round(pow(2, reader.q))):
            #     yield _ReaderSlot(self, i, Reader.State.QREP)
            yield _ReaderSlot(self, 0, Reader.State.QUERY)
            reader.qadjust_subround = False
            for i in range(1, round(pow(2, reader.q))):
                if reader.state == Reader.State.QADJUST:
                    reader.qadjust_subround = True
                    yield _ReaderSlot(self, i, Reader.State.QADJUST)
                    for j in range(1, round(pow(2, reader.q))):
                        yield _ReaderSlot(self, j, Reader.State.QREP)
                    break
                else:    
                    yield _ReaderSlot(self, i, Reader.State.QREP)
            slots = round(pow(2, reader.q))

            k = (reader.collision_slots / ((4.344 * slots - 16.28) + ((slots / (-2.282 - 0.273 * slots)) * reader.collision_slots))) + 0.2407 * np.log(slots + 42.56)
            
            l = (1.2592 + 1.513 * slots) * np.tan(1.234 * pow(slots, -0.9907) * reader.collision_slots)

            n = k * reader.single_slots + l

            new_q = int(np.floor(np.log2(n)))
            
            if new_q != reader.q:
                reader.q = new_q
           

            


        self._reader = reader
        self._slots = slots_gen()
        self._slot = None

    @property
    def index(self):
        return self._index

    @property
    def slot(self):
        return self._slot

    def next_slot(self):
        self._slot = next(self._slots)
        return self._slot


class Reader:

    class State(enum.Enum):
        OFF = _ReaderOFF()
        QUERY = _ReaderQuery()
        QREP = _ReaderQREP()
        QADJUST = _ReaderQADJUST()
        ACK = _ReaderACK()

        def __init__(self, obj):
            self.__obj__ = obj

        def __str__(self):
            return self.__obj__.__str__()

        def __getattr__(self, item):
            children = {'enter',
                        'get_timeout',
                        'handle_turn_on',
                        'handle_turf_off',
                        'handle_query_reply', 
                        'handle_timeout',
                        'handle_ack_reply',
                        'handle_query_adjust'
                        }
            if item in children:
                return getattr(self.__obj__, item)
            else:
                raise AttributeError
            
    # PIE time settings
    tari = 6.25e-6
    rtcal = 6.25e-6 * 3
    trcal = 6.25e-6 * 5
    delim = 12.5e-6

    temp = std.TempRange.NOMINAL

    #QADJUST subround flag
    qadjust_subround = False

    #slot state counters
    collision_slots = 0
    empty_slots = 0
    single_slots = 0

    # Round settings
    q = 10
    upDn = std.UpDn.NO_CHANGE
    tag_encoding = None
    trext = False
    dr = std.DivideRatio.DR_8
    sel = std.SelFlag.ALL
    session = std.Session.S0
    target = std.InventoryFlag.A

    # Operation modes
    read_tid_bank = False
    read_tid_words_num = None
    epc_bank = []
    
    def __init__(self, kernel=None):
        self.kernel = kernel
        self._state = Reader.State.OFF

        # Temporary data, e.g. received from the tag
        self.last_rn = None

        # Rounds managers
        self._round = None
        self._round_index = itertools.count()

    @property
    def state(self):
        return self._state

    def set_state(self, new_state):
        self._state = new_state
        return new_state.enter(self)

    @property
    def preamble(self):
        return std.ReaderPreamble(self.tari, self.rtcal, self.trcal, self.delim)
    
    @property
    def sync(self):
        return std.ReaderSync(self.tari, self.rtcal, self.delim)
    
    def receive(self, tag_frame):
        assert isinstance(tag_frame, std.TagFrame)
        reply = tag_frame.reply
        if isinstance(reply, std.QueryReply):
            return self._state.handle_query_reply(self, tag_frame)
        elif isinstance(reply, std.AckReply):
            return self._state.handle_ack_reply(self, tag_frame)
        else:
            raise ValueError('unexpected tag reply {}'.format(str(reply)))

    def timeout(self):
        return self._state.handle_timeout(self)
    
    def turn_on(self):
        return self._state.handle_turn_on(self)

    def turn_off(self):
        self._time_last_turned_on = None
        return self._state.handle_turn_off(self)
    
    # Round management

    @property
    def inventory_round(self):
        return self._round

    def stop_round(self):
        self._round = None

    def next_slot(self):
        if self._round is None:
            self._round = _ReaderRound(self, next(self._round_index))
            slot = self._round.next_slot()
        else:
            try:
                slot = self._round.next_slot()
            except StopIteration:
                self._round = _ReaderRound(self, next(self._round_index))
                slot = self._round.next_slot()
        return slot
    

class Tag:
    class State(enum.Enum):
        OFF = 0
        READY = 1
        ARBITRATE = 2
        REPLY = 3
        ACKNOWLEDGED = 4
        SECURED = 5
    
    # EPC Std. settings
    epc = ""            # should be a hex-string
    tid = None          # should be either None or hex-string
    user_mem = None     # should be either None or hex-string
    epc_bitlen = 96
    tid_bitlen = 64
    epc_prefix = 'AAAA'
    tid_prefix = 'AAAA'

    def __init__(self, tag_id):
        self._tag_id = tag_id

        # Internal registers and flags
        self._state = Tag.State.OFF
        self._slot_counter = 0
        self._q = 0
        self._rn = 0
        self._sl = False
        self._active_session = None
        self._preamble = None
        self.sessions = {}
        for session in {std.Session.S0, std.Session.S1, std.Session.S2,
                        std.Session.S3}:
            self.sessions[session] = None

        self._banks = {std.MemoryBank.EPC: lambda tag: tag.epc,
                       std.MemoryBank.TID: lambda tag: tag.tid,
                       std.MemoryBank.USER: lambda tag: tag.user_mem,
                       std.MemoryBank.RESERVED: lambda tag: None}

        # Parameters received from the reader
        self._encoding = std.TagEncoding.FM0
        self._blf = std.get_blf(std.DivideRatio.DR_8, 12.5e-6*6)
        self._trext = False


    @property
    def encoding(self):
        return self._encoding

    @property
    def blf(self):
        return self._blf

    @property
    def trext(self):
        return self._trext

    @property
    def tag_id(self):
        return self._tag_id

    @property
    def state(self):
        return self._state

    @property
    def slot_counter(self):
        return self._slot_counter

    @property
    def q(self):
        return self._q

    @property
    def rn(self):
        return self._rn

    @property
    def sl(self):
        return self._sl

    @property
    def s0(self):
        return self.sessions[std.Session.S0]

    @property
    def s1(self):
        return self.sessions[std.Session.S1]

    @property
    def s2(self):
        return self.sessions[std.Session.S2]

    @property
    def s3(self):
        return self.sessions[std.Session.S3]

    def setup(self):
        for session in {std.Session.S0, std.Session.S1, std.Session.S2,
                        std.Session.S3}:
            self.sessions[session] = std.InventoryFlag.A
        self._sl = False
        self._rn = 0x0000
        self._slot_counter = 0
        self._q = 0
        self._active_session = None
        self._preamble = None
        self._set_state(Tag.State.OFF)

    def _power_on(self):
        self._set_state(Tag.State.READY)

    def _power_off(self): 
        self._set_state(Tag.State.OFF)

    def _set_state(self, new_state):
        self._state = new_state

    def process_query(self, query):
        assert isinstance(query, std.ReaderFrame)
        assert isinstance(query.command, std.Query)
        assert isinstance(query.preamble, std.ReaderPreamble)
        command, preamble = query.command, query.preamble

        if self.state is Tag.State.OFF:
            return None

        if self.state not in {Tag.State.READY, Tag.State.ARBITRATE,
                              Tag.State.REPLY}:
            flag = self.sessions[self._active_session]
            self.sessions[self._active_session] = flag.invert()

        # Checking flags
        if not (self.sessions[command.session] == command.target and
                command.sel.match(self.sl)):
            self._set_state(Tag.State.READY)
            return None
        
        # Processing QUERY: select random slot, move to REPLY or ARBITRATE,
        # set current session, extract TRext and compute BLF
        self._active_session = command.session
        self._trext = command.trext
        self._encoding = command.m
        self._q = command.q
        self._slot_counter = np.random.randint(0, pow(2, command.q))
        self._preamble = std.create_tag_preamble(self.encoding, self.trext)
        if self._slot_counter == 0:
            self._set_state(Tag.State.REPLY)
            self._rn = np.random.randint(0, 0x10000)
            return std.TagFrame(self._preamble, std.QueryReply(self._rn))
        else:
            self._set_state(Tag.State.ARBITRATE)
            return None

    def process_query_rep(self, frame):
        assert isinstance(frame, std.ReaderFrame)
        assert isinstance(frame.command, std.QueryRep)
        qrep = frame.command
        if self.state is Tag.State.OFF:
            return None

        if qrep.session is not self._active_session:
            return None

        self._slot_counter -= 1
        if self._slot_counter == 0 and self.state is Tag.State.ARBITRATE:
            self._set_state(Tag.State.REPLY)
            self._rn = np.random.randint(0, 0x10000)
            return std.TagFrame(self._preamble, std.QueryReply(self._rn))
        else:
            if self.state in {Tag.State.ARBITRATE, Tag.State.REPLY}:
                self._set_state(Tag.State.ARBITRATE)
            elif self.state is not Tag.State.READY:
                flag = self.sessions[self._active_session]
                self.sessions[self._active_session] = flag.invert()
                self._set_state(Tag.State.READY)

            return None

    def process_query_adjust(self, frame):
        assert isinstance(frame, std.ReaderFrame)
        assert isinstance(frame.command, std.QueryAdjust)
        qadjust = frame.command
        if self.state is Tag.State.OFF:
            return None

        if qadjust.session is not self._active_session:
            return None
        
        self._slot_counter = np.random.randint(0, pow(2, self._q + qadjust.upDn.eval()))
        if self._slot_counter == 0 and self.state is Tag.State.ARBITRATE:
            self._set_state(Tag.State.REPLY)
            self._rn = np.random.randint(0, 0x10000)
            return std.TagFrame(self._preamble, std.QueryReply(self._rn))
        else:
            if self.state in {Tag.State.ARBITRATE, Tag.State.REPLY}:
                self._set_state(Tag.State.ARBITRATE)
            elif self.state is not Tag.State.READY:
                flag = self.sessions[self._active_session]
                self.sessions[self._active_session] = flag.invert()
                self._set_state(Tag.State.READY)

            return None

    def process_ack(self, frame):
        assert isinstance(frame, std.ReaderFrame)
        assert isinstance(frame.command, std.Ack)
        ack = frame.command
        if self.state is not Tag.State.REPLY:
            return None
        if ack.rn == self.rn:
            self._set_state(Tag.State.ACKNOWLEDGED)
            return std.TagFrame(self._preamble, std.AckReply(self.epc))
        else:
            self._set_state(Tag.State.ARBITRATE)
            return None

    def receive(self, frame):
        assert isinstance(frame, std.ReaderFrame)
        cmd = frame.command
        if isinstance(cmd, std.Query):
            return self.process_query(frame)
        elif isinstance(cmd, std.QueryRep):
            return self.process_query_rep(frame)
        elif isinstance(cmd, std.QueryAdjust):
            return self.process_query_adjust(frame)
        elif isinstance(cmd, std.Ack):
            return self.process_ack(frame)
        else:
            raise TypeError("unexpected command '{}'".format(frame))

    @staticmethod
    def get_new_slot_state(new_slot_counter, curr_state, match_flags=True):
        if curr_state is Tag.State.OFF:
            return Tag.State.OFF
        else:
            if not match_flags:
                return Tag.State.READY
            else:
                if new_slot_counter == 0:
                    return Tag.State.REPLY
                else:
                    return Tag.State.ARBITRATE
    
#############################################################################
# Generators
#############################################################################
class Generator:
    epc_bitlen = 96
    tid_bitlen = 64
    epc_prefix = 'AAAA'
    tid_prefix = 'AAAA'

    def __init__(self):
        def hex_string_bitlen(s):
            return len(s.strip()) * 4
        epc_suffix_bitlen = self.epc_bitlen - hex_string_bitlen(self.epc_prefix)
        tid_suffix_bitlen = self.tid_bitlen - hex_string_bitlen(self.tid_prefix)
        self._epc_suffix = '0' * int(np.ceil(epc_suffix_bitlen / 4))
        self._tid_suffix = '0' * int(np.ceil(tid_suffix_bitlen / 4))

    def create_tag(self, tag_counter):
        tag_id = tag_counter
        tag = Tag(tag_id)
        tag.epc = self.epc_prefix + self._epc_suffix
        tag.tid = self.tid_prefix + self._tid_suffix
        self._epc_suffix = inc_hex_string(self._epc_suffix)
        self._tid_suffix = inc_hex_string(self._tid_suffix)
        return tag


def inc_hex_string(s):
    pos = len(s) - 1
    while pos >= 0:
        x = int(s[pos], 16)
        if x < 15:
            return s[:pos] + "{:1X}".format(x + 1) + ("0" * (len(s) - 1 - pos))
        else:
            pos -= 1
    return "0" * len(s)

#############################################################################
# Transactions
#############################################################################

class Transaction(object):
    timeout_event_id = None
    response_start_event_id = None

    def __init__(self, reader, command, replies, time):
        self._command = command
        self._reader = reader
        self._replies = tuple(replies)
        self._start_time = time

        self._command_duration = command.duration
        self._command_end_time = time + self._command_duration

        # Whether the transaction involves replies, its duration is measured
        # as maximum of (command + T4) and (command + T1 + reply + T3).
        # T4 is a minimum interval between successive commands.
        # Reply duration is taken as maximum reply duration among all replies.
        # If no replies exist, reader.state.get_timeout() is used.

        if replies:
            reply_durations = [f.get_duration(tag.blf) for tag, f in replies]
            self._reply_duration = np.max(reply_durations)
            t1 = std.link_t1_min(
                reader.rtcal, reader.trcal, reader.dr, reader.temp)
            t2 = std.link_t2_max(reader.trcal, reader.dr)   # NOTE: may be min?
            t4 = std.link_t4(reader.rtcal)
            exchange_duration = (self._command_duration + t1 +
                                 self._reply_duration + t2)
            self._reply_start_time = time + t1
            self._reply_end_time = self._reply_start_time + self._reply_duration
            self._duration = max(exchange_duration, self._command_duration + t4)
        else:
            self._reply_start_time = None
            self._reply_end_time = None
            self._reply_duration = None
            self._duration = reader.state.get_timeout(reader)

        self._finish_time = time + self._duration

    @property
    def command(self):
        return self._command

    @property
    def reader(self):
        return self._reader

    @property
    def replies(self):
        return self._replies

    @property
    def start_time(self):
        return self._start_time

    @property
    def command_end_time(self):
        return self._command_end_time

    @property
    def reply_start_time(self):
        return self._reply_start_time

    @property
    def reply_end_time(self):
        return self._reply_end_time

    @property
    def duration(self):
        return self._duration

    @property
    def finish_time(self):
        return self._finish_time

    @property
    def command_duration(self):
        return self._command_duration

    @property
    def reply_duration(self):
        return self._reply_duration

    @property
    def tags(self):
        return (tag for (tag, _) in self.replies)

    @property
    def reader_rx_power_map(self):
        return self._reader_rx_powers

    def received_tag_frame(self):
        # NOTE: if two or more tags reply, their reply is treated as collision
        #       no matter of SNR. Try to implement this.
        
        if len(self.replies) == 0:
            self._reader.empty_slots += 1
            # self._reader.upDn = std.UpDn.DECREASE
            # self._reader.set_state(Reader.State.QADJUST)
            # self._reader._state.handle_query_adjust(self._reader)
            return None, None
        
        if len(self.replies) > 1:
            self._reader.collision_slots += 1
            # if not self.reader.qadjust_subround:
            #     print('collision!')
            #     self._reader.upDn = std.UpDn.INCREASE
            #     self._reader.set_state(Reader.State.QADJUST)
            #     self._reader._state.handle_query_adjust(self._reader)
            return None, None

        if isinstance(self.replies[0][1].reply, std.QueryReply):
            self._reader.single_slots += 1

        tag, frame = self.replies[0]

        return (tag, frame) 

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


def start_simulation(kernel):
    assert isinstance(kernel.context, Model)
    ctx = kernel.context
    ctx.reader.kernel = kernel
    kernel.call(turn_reader_on, ctx.reader)


def turn_reader_on(kernel, reader):
    ctx = kernel.context

    cmd_frame = reader.turn_on()

    # Processing new command (reader frame)
    transaction = build_transaction(kernel, reader, cmd_frame)
    ctx.transaction = transaction
    ctx.transaction.timeout_event_id = kernel.schedule(
        transaction.duration, finish_transaction, transaction)


def build_transaction(kernel, reader, reader_frame):
    ctx = kernel.context
    response = ((tag, tag.receive(reader_frame)) for tag in ctx.tags)
    tag_frames = [(tag, frame) for (tag, frame) in response
                  if frame is not None]
    now = kernel.time
    #print(now)
    trans = Transaction(reader, reader_frame, tag_frames, now)
    #print(trans.command.command)
    return trans

def finish_transaction(kernel, transaction):
    ctx = kernel.context
    reader = ctx.reader
    assert transaction is ctx.transaction

    tag, frame = transaction.received_tag_frame()

    if frame is not None:
        cmd_frame = reader.receive(frame)
    else:
        cmd_frame = reader.timeout()
    
    # Processing new command (reader frame)
    if len(ctx.reader.epc_bank) == ctx.max_tags_num:
        print(kernel.time)
        return
    ctx.transaction = build_transaction(kernel, reader, cmd_frame)
    ctx.transaction.timeout_event_id = kernel.schedule(
        transaction.duration, finish_transaction, ctx.transaction)
    #if isinstance(kernel.context.transaction.command.command, std.Query):
    #    print(kernel.context.transaction.command.command)


def simulate_tags():

    # 0) Building the model
    model = Model()
    model.max_tags_num = 200

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






# def build_transaction(reader, reader_frame, tags):
#     response = ((tag, tag.receive(reader_frame)) for tag in tags)
#     tag_frames = [(tag, frame) for (tag, frame) in response
#                   if frame is not None]
#     return Transaction(reader, reader_frame, tag_frames, 0)

# def finish_transaction(reader, transaction):
#     tag, tag_frame = transaction.received_tag_frame()
#     if tag_frame is not None:
#         return reader.receive(tag_frame)
#     else:
#         return reader.timeout()

# reader = Reader()
# tag_generator = Generator()
# tags = []

# for i in range(0, 4):
#     tag = tag_generator.create_tag(i)
#     tag.setup()
#     tag._power_on()
#     tags.append(tag)

# cmd_frame = reader.turn_on()

# while len(reader.epc_bank) < 4:
#     transaction = build_transaction(reader, cmd_frame, tags)
#     cmd_frame = finish_transaction(reader, transaction)
#     print(cmd_frame)




