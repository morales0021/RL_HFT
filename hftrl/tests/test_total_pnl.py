from futsimulator.positions.position import SideOrder
from futsimulator.utils.performance import get_total_pnl

dic_info = {
    'open_orders': {
        'o_pnl': -0.03125,
        'total_size': 1,
        'av_o_price': 125.125,
        'cl_pnl': 0.0,
        'delta_t': 1.166323184967041,
        'total_orders': 1,
        'av_cl_price': 0.0,
        'side': SideOrder.buy,
        'takeprofit': [], 'stoploss': [],
        'opened': [{'id_order': 1, 'open_price': 125.125, 'size': 1}]
        },
        'closed_orders': {}, 'stop_orders': [], 'limit_orders': []
    }


cl_pnl = get_total_pnl(dic_info, closed = True)
o_pnl = get_total_pnl(dic_info, closed = False)

print(cl_pnl)
print(o_pnl)