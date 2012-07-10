def myint32(val):
    val = max(-2**31+1,val)
    val = min(2**31-1,val)
    return int(val)

def myint16(val):
    val = max(-2**15+1,val)
    val = min(2**15-1,val)
    return int(val)
