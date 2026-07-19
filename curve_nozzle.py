import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 設計パラメータ
# -----------------------------
# 半径 [mm]
R_in = 17.0      # 上流半径
R_t  = 9.14      # スロート半径
epsilon = 3.17   # 開口比 A_e / A_t

# 出口半径 [mm]
R_e = R_t * np.sqrt(epsilon)

# 軸方向長さ [mm]
L_c = 15.0       # 収束部長さ
L_d = 25.0       # 拡大部長さ
L   = L_c + L_d  # 全長

# 壁面角度 [deg]
theta_t_deg = 30.0   # スロートでの壁角度（収束→拡大の接線）
theta_e_deg = 5.0    # 出口での壁角度

theta_t = np.deg2rad(theta_t_deg)
theta_e = np.deg2rad(theta_e_deg)

# 接線（dr/dx）
drdx_inlet  = 0.0             # 上流側はほぼ平行流
drdx_throat = np.tan(theta_t)
drdx_exit   = np.tan(theta_e)

# -----------------------------
# Hermite基底関数
# -----------------------------
def hermite_basis(s):
    """
    3次Hermite基底関数 h0, h1, h2, h3 を返す
    s: 0〜1 の無次元座標
    """
    h0 =  2*s**3 - 3*s**2 + 1
    h1 = -2*s**3 + 3*s**2
    h2 =  s**3 - 2*s**2 + s
    h3 =  s**3 - s**2
    return h0, h1, h2, h3

# -----------------------------
# 収束部の半径分布 r_c(x)
# -----------------------------
def radius_converging(x):
    """
    収束部の半径 r(x) を返す
    x: 0〜L_c [mm]
    """
    s = x / L_c  # 無次元座標
    h0, h1, h2, h3 = hermite_basis(s)
    # r'(0) = drdx_inlet, r'(L_c) = drdx_throat
    r = (h0 * R_in
         + h1 * R_t
         + h2 * L_c * drdx_inlet
         + h3 * L_c * drdx_throat)
    return r

# -----------------------------
# 拡大部の半径分布 r_d(x)
# -----------------------------
def radius_diverging(x):
    """
    拡大部の半径 r(x) を返す
    x: L_c〜L_c+L_d [mm]
    """
    t = (x - L_c) / L_d  # 無次元座標
    h0, h1, h2, h3 = hermite_basis(t)
    # r'(L_c) = drdx_throat, r'(L_c+L_d) = drdx_exit
    r = (h0 * R_t
         + h1 * R_e
         + h2 * L_d * drdx_throat
         + h3 * L_d * drdx_exit)
    return r

# -----------------------------
# ノズル形状のサンプリング
# -----------------------------
# 収束部
x_c = np.linspace(0.0, L_c, 200)
r_c = radius_converging(x_c)

# 拡大部
x_d = np.linspace(L_c, L_c + L_d, 400)
r_d = radius_diverging(x_d)

# -----------------------------
# 図示
# -----------------------------
plt.figure(figsize=(8, 4))

# 軸対称ノズルなので、半径を上下に描いてみる
plt.plot(x_c,  r_c,  'b', label='Converging')
plt.plot(x_c, -r_c,  'b')
plt.plot(x_d,  r_d,  'r', label='Diverging')
plt.plot(x_d, -r_d,  'r')

plt.axvline(L_c, color='k', linestyle='--', alpha=0.5, label='Throat position') # なんかthroatずれてる

plt.xlabel('Axial coordinate x [mm]')
plt.ylabel('Radius r [mm]')
plt.title('Supersonic nozzle contour (Hermite smooth wall)')
plt.legend()
plt.grid(True)
plt.axis('equal')  # 形状を正しく見せるために縦横比を揃える

plt.tight_layout()
plt.show()

# -----------------------------
# 参考: 幾何情報の表示
# -----------------------------
print(f"R_in  = {R_in:.3f} mm")
print(f"R_t   = {R_t:.3f} mm")
print(f"R_e   = {R_e:.3f} mm (epsilon = {epsilon})")
print(f"L_c   = {L_c:.3f} mm")
print(f"L_d   = {L_d:.3f} mm")
print(f"L_tot = {L:.3f} mm")
print(f"theta_t = {theta_t_deg:.1f} deg, theta_e = {theta_e_deg:.1f} deg")

# throatがずれたので対応
min = np.min(r_c)
print(min)

# CSVに書き出し
x_wall = np.concatenate([x_c, x_d])
r_wall = np.concatenate([r_c, r_d])
data = np.column_stack([x_wall, r_wall])
np.savetxt("nozzle_contour_2.csv", data, delimiter=",",
           header="x_mm,r_mm", comments="")
