
proto = None
Transaction = None

def set_transaction_file(protocol):
    global proto
    global Transaction
    proto = protocol
    match protocol:
        case 'PEFSA':
            from objects.pefsa import Transaction
        case 'CHEN':
            from objects.chen import Transaction
        case 'ADAPTIVE':
            from objects.adaptive import Transaction
        case 'FAST':
            from objects.fast import Transaction
        case 'ILCM':
            from objects.ilcm import Transaction
        case 'SUBEP':
            from objects.subep import Transaction


def start_simulation(kernel):
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
    trans = Transaction(reader, reader_frame, tag_frames, now)
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
        print(ctx.reader.all_slots)
        return
    ctx.transaction = build_transaction(kernel, reader, cmd_frame)
    ctx.transaction.timeout_event_id = kernel.schedule(
        transaction.duration, finish_transaction, ctx.transaction)