"""
NYC TLC 原始黄色出租车 (Parquet, LocationID格式) → ST-SSL 格式

原始数据: 2015年1月 + 2月 (60天), 与 STDN/ST-SSL 论文一致
窗口: 35 = 8 recent + 9×3 daily (来自 ST-SSL BJTaxi 配置注释)

用法: python generate_stssl_data.py
"""
import os, sys, time
import numpy as np
import pandas as pd

# ==================== 修改这里的路径 ====================
RAW_FILES = [
    r'D:\0BigData\22Data_Warehousing_and_Data_Mining\code\traffic_pred\data\raw\yellow_tripdata_2015-01.parquet',
    r'D:\0BigData\22Data_Warehousing_and_Data_Mining\code\traffic_pred\data\raw\yellow_tripdata_2015-02.parquet',
]
OUTPUT_DIR = r'D:\0BigData\22Data_Warehousing_and_Data_Mining\code\traffic_pred\data\processed\NYCTaxi_ours'


# ==================== 网格配置 ====================
GRID_ROWS, GRID_COLS = 10, 20
NUM_NODES = GRID_ROWS * GRID_COLS
LAT_MIN, LAT_MAX = 40.60, 40.90
LON_MIN, LON_MAX = -74.06, -73.77
TIME_INTERVAL = 30
STEPS_PER_DAY = 48

START_DATE = pd.Timestamp('2015-01-01')
END_DATE   = pd.Timestamp('2015-03-01')
TOTAL_DAYS = 60
TOTAL_STEPS = TOTAL_DAYS * STEPS_PER_DAY  # 2880

# 窗口 35 = 8 + 9×3
NUM_RECENT = 8
NUM_DAYS_BACK = 3
WINDOW_PER_DAY = 9
INPUT_LEN = NUM_RECENT + NUM_DAYS_BACK * WINDOW_PER_DAY  # 35
TRAIN_RATIO, VAL_RATIO = 0.7, 0.1

# ==================== NYC Taxi Zone → (lat, lon) 质心映射 ====================
# 来源: NYC TLC taxi_zones shapefile 质心 (WGS84)
# LocationID → (latitude, longitude)
ZONE_CENTROIDS = {
    1:(40.6951,-74.1749),2:(40.6569,-73.7825),3:(40.6690,-73.8504),4:(40.7268,-73.9816),
    5:(40.6979,-73.7676),6:(40.6564,-73.7553),7:(40.7640,-73.9237),8:(40.6591,-73.7600),
    9:(40.6694,-73.7370),10:(40.7111,-73.9411),11:(40.6814,-73.8408),12:(40.7056,-74.0178),
    13:(40.7112,-74.0150),14:(40.6350,-74.0284),15:(40.6700,-73.9740),16:(40.6768,-73.9545),
    17:(40.6280,-73.9439),18:(40.6530,-73.8791),19:(40.6800,-73.8981),20:(40.6722,-73.8694),
    21:(40.8113,-73.9553),22:(40.7025,-73.9500),23:(40.6694,-73.8954),24:(40.7937,-73.9677),
    25:(40.6397,-73.8817),26:(40.7420,-73.8607),27:(40.6779,-73.9056),28:(40.7400,-73.9030),
    29:(40.7550,-73.8810),30:(40.8680,-73.9170),31:(40.6891,-73.9754),32:(40.6520,-73.7517),
    33:(40.7098,-73.9412),34:(40.6529,-73.9310),35:(40.6582,-73.9627),36:(40.8580,-73.9280),
    37:(40.7102,-73.9217),38:(40.6770,-73.7840),39:(40.6870,-73.9954),40:(40.6840,-73.9517),
    41:(40.8146,-73.9408),42:(40.8231,-73.9447),43:(40.7754,-73.9709),44:(40.7590,-73.9700),
    45:(40.7461,-73.9960),46:(40.6864,-73.7850),47:(40.7280,-73.9960),48:(40.7600,-73.9880),
    49:(40.7240,-73.9740),50:(40.7640,-73.9890),51:(40.7390,-73.8260),52:(40.7010,-73.7620),
    53:(40.6756,-73.9490),54:(40.6721,-73.7310),55:(40.6581,-73.7360),56:(40.6843,-73.7400),
    57:(40.6757,-73.7880),58:(40.7240,-73.7790),59:(40.7420,-73.8520),60:(40.6844,-73.7970),
    61:(40.6738,-73.9588),62:(40.6680,-73.9620),63:(40.6790,-73.9650),64:(40.6766,-73.9310),
    65:(40.6400,-73.9540),66:(40.7280,-73.9880),67:(40.7380,-73.8690),68:(40.7401,-73.9997),
    69:(40.6530,-73.9520),70:(40.6498,-73.9570),71:(40.6770,-73.9020),72:(40.6560,-73.7440),
    73:(40.6390,-73.7470),74:(40.7297,-73.9773),75:(40.7620,-73.9250),76:(40.7370,-73.8780),
    77:(40.7310,-73.8530),78:(40.6530,-73.7620),79:(40.7269,-73.9832),80:(40.6716,-73.8800),
    81:(40.6630,-73.8580),82:(40.6940,-73.7880),83:(40.6589,-73.8024),84:(40.6500,-73.9630),
    85:(40.6660,-73.7190),86:(40.7355,-73.9770),87:(40.7076,-74.0079),88:(40.7048,-74.0134),
    89:(40.6580,-73.7660),90:(40.7420,-73.9916),91:(40.6360,-73.8830),92:(40.6490,-73.9590),
    93:(40.6380,-73.7570),94:(40.7540,-73.8430),95:(40.7350,-73.8330),96:(40.7580,-73.8270),
    97:(40.6537,-73.9641),98:(40.6500,-73.9350),99:(40.7360,-73.8200),100:(40.7539,-73.9904),
    101:(40.7320,-73.8670),102:(40.7300,-73.8450),103:(40.6804,-73.7690),104:(40.7360,-73.8110),
    105:(40.6960,-73.7760),106:(40.6610,-73.7540),107:(40.7400,-73.9821),108:(40.7290,-73.8580),
    109:(40.7270,-73.8460),110:(40.6900,-73.7780),111:(40.7170,-73.8170),112:(40.6640,-73.8910),
    113:(40.7360,-74.0020),114:(40.7310,-74.0010),115:(40.6840,-73.8040),116:(40.8033,-73.9360),
    117:(40.6410,-73.7630),118:(40.6850,-73.7750),119:(40.6600,-73.8460),120:(40.8290,-73.9330),
    121:(40.6900,-73.9930),122:(40.8550,-73.9200),123:(40.7300,-73.9510),124:(40.8270,-73.9470),
    125:(40.8221,-73.9479),126:(40.6570,-73.7660),127:(40.6510,-73.7520),128:(40.8147,-73.9372),
    129:(40.7489,-73.8824),130:(40.6625,-73.8600),131:(40.7550,-73.8450),132:(40.6474,-73.7855),
    133:(40.7110,-73.7940),134:(40.7620,-73.9494),135:(40.7670,-73.9120),136:(40.7790,-73.9534),
    137:(40.7470,-73.9780),138:(40.7700,-73.8730),139:(40.7340,-73.8700),140:(40.7664,-73.9566),
    141:(40.7706,-73.9618),142:(40.7750,-73.9830),143:(40.7810,-73.9830),144:(40.7213,-73.9969),
    145:(40.7430,-73.9550),146:(40.6473,-73.8860),147:(40.8350,-73.9250),148:(40.7170,-73.9870),
    149:(40.7560,-73.8100),150:(40.7450,-73.8160),151:(40.7997,-73.9669),152:(40.8130,-73.9577),
    153:(40.8740,-73.9110),154:(40.7310,-73.9130),155:(40.6510,-73.7460),156:(40.7050,-73.7430),
    157:(40.7360,-73.7630),158:(40.7399,-74.0085),159:(40.7790,-73.9350),160:(40.7280,-73.7920),
    161:(40.7548,-73.9827),162:(40.7548,-73.9705),163:(40.7649,-73.9820),164:(40.7508,-73.9847),
    165:(40.7910,-73.9440),166:(40.8069,-73.9589),167:(40.7100,-73.7720),168:(40.7250,-73.8050),
    169:(40.7670,-73.9190),170:(40.7490,-73.9766),171:(40.6780,-73.8640),172:(40.8405,-73.9406),
    173:(40.7120,-73.7860),174:(40.6640,-73.8430),175:(40.8240,-73.9420),176:(40.6870,-73.8220),
    177:(40.6780,-73.8890),178:(40.6590,-73.8270),179:(40.6640,-73.7530),180:(40.6520,-73.8720),
    181:(40.6700,-73.9810),182:(40.6530,-73.9410),183:(40.6390,-73.9320),184:(40.6300,-73.7640),
    185:(40.7160,-73.8120),186:(40.7498,-73.9937),187:(40.6630,-73.7290),188:(40.6789,-73.9694),
    189:(40.6730,-73.9220),190:(40.6700,-73.8560),191:(40.7470,-73.8650),192:(40.6910,-73.8090),
    193:(40.7310,-73.8560),194:(40.7920,-73.9210),195:(40.6750,-73.9950),196:(40.7460,-73.8210),
    197:(40.7530,-73.7750),198:(40.7110,-73.8110),199:(40.7430,-73.8330),200:(40.7070,-73.8200),
    201:(40.8200,-73.9500),202:(40.7591,-73.9521),203:(40.7250,-73.8290),204:(40.6090,-74.0680),
    205:(40.5960,-74.0660),206:(40.5920,-74.1420),207:(40.6350,-74.0120),208:(40.6350,-74.1490),
    209:(40.7249,-74.0029),210:(40.6260,-74.0760),211:(40.6410,-74.0960),212:(40.6390,-74.1220),
    213:(40.5550,-74.1740),214:(40.6420,-74.0760),215:(40.7390,-73.9200),216:(40.5730,-74.1190),
    217:(40.6090,-74.1340),218:(40.5940,-74.0870),219:(40.7310,-73.7560),220:(40.6250,-74.1650),
    221:(40.5890,-74.0960),222:(40.6510,-74.0050),223:(40.7200,-73.9440),224:(40.7369,-73.9764),
    225:(40.6440,-73.9000),226:(40.6560,-74.0020),227:(40.7280,-73.8060),228:(40.7490,-73.8550),
    229:(40.7570,-73.9630),230:(40.7580,-73.9870),231:(40.7168,-74.0093),232:(40.7220,-74.0083),
    233:(40.7578,-73.9683),234:(40.7350,-73.9906),235:(40.6530,-74.0070),236:(40.7825,-73.9486),
    237:(40.7695,-73.9573),238:(40.7870,-73.9730),239:(40.7780,-73.9800),240:(40.6520,-74.0260),
    241:(40.6830,-73.7650),242:(40.7120,-73.7840),243:(40.8480,-73.9320),244:(40.8370,-73.9410),
    245:(40.6270,-74.1100),246:(40.7500,-74.0010),247:(40.8290,-73.9240),248:(40.8300,-73.8800),
    249:(40.7354,-74.0057),250:(40.8290,-73.8570),251:(40.6160,-74.1390),252:(40.7940,-73.8080),
    253:(40.7540,-73.8410),254:(40.8850,-73.8640),255:(40.7170,-73.9580),256:(40.7090,-73.9580),
    257:(40.6530,-73.9760),258:(40.6900,-73.8580),259:(40.9020,-73.8640),260:(40.7440,-73.9020),
    261:(40.7118,-74.0122),262:(40.7758,-73.9493),263:(40.7728,-73.9557),264:(40.7430,-73.9720),
    265:(40.7430,-73.9720),
}

def log(msg):
    print(msg, flush=True)


def latlon_to_node(lat, lon):
    """(lat, lon) → 网格节点 index, 超出范围返回 -1"""
    r = int((lat - LAT_MIN) / (LAT_MAX - LAT_MIN) * GRID_ROWS)
    c = int((lon - LON_MIN) / (LON_MAX - LON_MIN) * GRID_COLS)
    if 0 <= r < GRID_ROWS and 0 <= c < GRID_COLS:
        return r * GRID_COLS + c
    return -1


# 预计算 LocationID → 网格节点 查找表
ZONE_TO_NODE = {}
for zid, (lat, lon) in ZONE_CENTROIDS.items():
    ZONE_TO_NODE[zid] = latlon_to_node(lat, lon)


# ==================== Step 1: 加载 ====================
def load_raw():
    log('=' * 60)
    log('Step 1: 加载原始数据')
    log('=' * 60)

    dfs = []
    for path in RAW_FILES:
        if not os.path.exists(path):
            log(f'  [ERROR] 文件不存在: {path}')
            sys.exit(1)
        log(f'  加载 {path} ...')
        t0 = time.time()
        df = pd.read_parquet(path)
        log(f'    {len(df):,} 行  ({time.time()-t0:.1f}s)')
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    log(f'  合并: {len(df):,} 行')

    # 统一列名
    col_map = {}
    for col in df.columns:
        cl = col.strip().lower()
        if 'pickup' in cl and ('time' in cl or 'datetime' in cl or 'date' in cl):
            col_map[col] = 'pickup_datetime'
        elif cl in ('pulocationid',):
            col_map[col] = 'PULocationID'
        elif cl in ('dolocationid',):
            col_map[col] = 'DOLocationID'
        elif 'pickup' in cl and ('lon' in cl or 'lng' in cl):
            col_map[col] = 'pickup_longitude'
        elif 'pickup' in cl and 'lat' in cl:
            col_map[col] = 'pickup_latitude'
        elif 'dropoff' in cl and ('lon' in cl or 'lng' in cl):
            col_map[col] = 'dropoff_longitude'
        elif 'dropoff' in cl and 'lat' in cl:
            col_map[col] = 'dropoff_latitude'
    df = df.rename(columns=col_map)
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

    # 判断数据格式
    has_latlon = 'pickup_longitude' in df.columns and 'pickup_latitude' in df.columns
    has_locid  = 'PULocationID' in df.columns and 'DOLocationID' in df.columns

    if has_latlon:
        log('  数据格式: 经纬度 (lat/lon)')
    elif has_locid:
        log('  数据格式: LocationID (需要映射到网格)')
    else:
        log(f'  [ERROR] 无法识别数据格式, 列: {list(df.columns)}')
        sys.exit(1)

    return df, has_latlon


# ==================== Step 2: 需求张量 ====================
def build_demand(df, has_latlon):
    log('\n' + '=' * 60)
    log('Step 2: 构建需求张量 (T, 200, 2)')
    log('=' * 60)

    mask = (df['pickup_datetime'] >= START_DATE) & (df['pickup_datetime'] < END_DATE)
    df = df[mask].copy()
    log(f'  时间范围内: {len(df):,}')

    if has_latlon:
        # 经纬度模式: 直接映射
        lat_ok = (df['pickup_latitude'].between(LAT_MIN, LAT_MAX) &
                  df['dropoff_latitude'].between(LAT_MIN, LAT_MAX))
        lon_ok = (df['pickup_longitude'].between(LON_MIN, LON_MAX) &
                  df['dropoff_longitude'].between(LON_MIN, LON_MAX))
        df = df[lat_ok & lon_ok].copy()

        def to_node_arr(lat, lon):
            r = ((lat - LAT_MIN) / (LAT_MAX - LAT_MIN) * GRID_ROWS).astype(int)
            c = ((lon - LON_MIN) / (LON_MAX - LON_MIN) * GRID_COLS).astype(int)
            v = (r >= 0) & (r < GRID_ROWS) & (c >= 0) & (c < GRID_COLS)
            return np.where(v, r * GRID_COLS + c, -1), v

        p_node, p_ok = to_node_arr(df['pickup_latitude'].values, df['pickup_longitude'].values)
        d_node, d_ok = to_node_arr(df['dropoff_latitude'].values, df['dropoff_longitude'].values)
        both = p_ok & d_ok
    else:
        # LocationID 模式: 查表映射
        pu_ids = df['PULocationID'].values.astype(int)
        do_ids = df['DOLocationID'].values.astype(int)

        p_node = np.array([ZONE_TO_NODE.get(z, -1) for z in pu_ids])
        d_node = np.array([ZONE_TO_NODE.get(z, -1) for z in do_ids])
        both = (p_node >= 0) & (d_node >= 0)

    log(f'  网格内有效: {both.sum():,} / {len(df):,} ({both.mean()*100:.1f}%)')

    p_node = p_node[both]
    d_node = d_node[both]
    dt = df['pickup_datetime'].values[both]

    slot = ((dt - np.datetime64(START_DATE)) / np.timedelta64(TIME_INTERVAL, 'm')).astype(int)
    valid = (slot >= 0) & (slot < TOTAL_STEPS)
    slot, p_node, d_node = slot[valid], p_node[valid], d_node[valid]

    log(f'  构建张量 ({TOTAL_STEPS}, {NUM_NODES}, 2) ...')
    demand = np.zeros((TOTAL_STEPS, NUM_NODES, 2), dtype=np.float32)
    np.add.at(demand[:, :, 0], (slot, p_node), 1)
    np.add.at(demand[:, :, 1], (slot, d_node), 1)

    log(f'  pickup: {demand[:,:,0].sum():.0f}, dropoff: {demand[:,:,1].sum():.0f}')
    return demand


# ==================== Step 3: 滑动窗口 ====================
def build_offsets():
    x_off = []
    half = WINDOW_PER_DAY // 2
    for day in range(NUM_DAYS_BACK, 0, -1):
        center = -day * STEPS_PER_DAY
        x_off.extend(range(center - half, center + half + 1))
    x_off.extend(range(-NUM_RECENT, 0))
    return np.array(x_off, dtype=int), np.array([1], dtype=int)


def generate_samples(demand, x_off, y_off):
    log('\n' + '=' * 60)
    log('Step 3: 生成滑动窗口样本')
    log('=' * 60)

    T, N, C = demand.shape
    t0 = -x_off.min()
    t1 = T - y_off.max()
    n = t1 - t0
    log(f'  样本数: {n}, 目标 t={t0}~{t1-1}')

    targets = np.arange(t0, t1)
    X = demand[targets[:, None] + x_off[None, :]]   # (n, 35, N, C)
    Y = demand[targets[:, None] + y_off[None, :]]   # (n, 1,  N, C)
    log(f'  X: {X.shape}, Y: {Y.shape}')
    return X, Y


# ==================== Step 4: 划分 ====================
def split_data(X, Y):
    log('\n' + '=' * 60)
    log('Step 4: 划分 train/val/test (7:1:2)')
    log('=' * 60)
    n = len(X)
    i1, i2 = int(n * TRAIN_RATIO), int(n * (TRAIN_RATIO + VAL_RATIO))
    splits = {'train': (X[:i1], Y[:i1]), 'val': (X[i1:i2], Y[i1:i2]), 'test': (X[i2:], Y[i2:])}
    for k, (x, y) in splits.items():
        log(f'  {k:5s}: x={x.shape}, y={y.shape}')
    log(f'  参考: train=1912, val=274, test=546')
    return splits


# ==================== Step 5: 邻接矩阵 ====================
def build_adj():
    log('\n' + '=' * 60)
    log('Step 5: 邻接矩阵')
    log('=' * 60)
    N = NUM_NODES
    adj = np.zeros((N, N), dtype=np.float32)
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            i = r * GRID_COLS + c
            adj[i, i] = 1.0
            if r > 0:            adj[i, (r-1)*GRID_COLS+c] = 1.0
            if r < GRID_ROWS-1:  adj[i, (r+1)*GRID_COLS+c] = 1.0
            if c > 0:            adj[i, r*GRID_COLS+(c-1)] = 1.0
            if c < GRID_COLS-1:  adj[i, r*GRID_COLS+(c+1)] = 1.0
    adj /= adj.sum(axis=1, keepdims=True)
    log(f'  shape: {adj.shape}, density: {(adj>0).sum()/adj.size:.4f}')
    return adj


# ==================== Step 6: 保存 ====================
def save_all(splits, adj, x_off, y_off):
    log('\n' + '=' * 60)
    log('Step 6: 保存')
    log('=' * 60)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for k, (x, y) in splits.items():
        p = os.path.join(OUTPUT_DIR, f'{k}.npz')
        np.savez_compressed(p, x=x, y=y, x_offsets=x_off, y_offsets=y_off)
        log(f'  {p}  ({os.path.getsize(p)/1024/1024:.1f} MB)')
    p = os.path.join(OUTPUT_DIR, 'adj_mx.npz')
    np.savez_compressed(p, adj_mx=adj)
    log(f'  {p}')


# ==================== 主流程 ====================
def main():
    t_all = time.time()
    log('=' * 60)
    log('NYC TLC → ST-SSL 格式')
    log(f'  窗口: {INPUT_LEN} = {NUM_RECENT} recent + {NUM_DAYS_BACK}x{WINDOW_PER_DAY} daily')
    log('=' * 60)

    df, has_latlon = load_raw()
    demand = build_demand(df, has_latlon)
    del df

    x_off, y_off = build_offsets()
    log(f'\n  X_offsets ({len(x_off)}步):')
    idx = 0
    for d in range(NUM_DAYS_BACK, 0, -1):
        log(f'    Day-{d}: {x_off[idx:idx+WINDOW_PER_DAY].tolist()}')
        idx += WINDOW_PER_DAY
    log(f'    Recent: {x_off[idx:].tolist()}')

    X, Y = generate_samples(demand, x_off, y_off)
    del demand
    splits = split_data(X, Y)
    del X, Y
    adj = build_adj()
    save_all(splits, adj, x_off, y_off)

    log(f'\n{"="*60}')
    log(f'完成! 耗时 {time.time()-t_all:.1f}s')
    log(f'输出: {OUTPUT_DIR}/')
    log(f'{"="*60}')

if __name__ == '__main__':
    main()