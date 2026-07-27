#+----------------------------------------------------------------------------+
#|                                                periapsis_analysis_model.py |
#|          Copyright 2022-2025 HP Investment Trading and Gambling Strategies |
#|                                                        https://hp-fx-g.com |
#+----------------------------------------------------------------------------+

##--- import modules
import os
import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.ticker as mtick
from matplotlib.collections import LineCollection
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from adjustText import adjust_text
from matplotlib.patches import Rectangle
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.cluster import KMeans
from scipy.signal import argrelextrema
import matplotlib.patheffects as path_effects
from matplotlib.lines import Line2D

#+----------------------------------------------------------------------------+
#| @func: get parent directory                                                |
#| @desc: finds and returns the parent directory of the current script        |
#| @params: N/A                                                               |
#| @return: parent dir[ectory] --> parent directory of the current script     |
#+----------------------------------------------------------------------------+
def get_parent_directory():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    return parent_dir


#+----------------------------------------------------------------------------+
#| @class: Periapsis Theory Modeler                                           |
#| @desc: applies the periapsis volume increase theory to a trading system    |
#| @params: N/A                                                               |
#| @return: N/A                                                               |
#+----------------------------------------------------------------------------+
class Periapsis_Theory_Base_Modeler():
    ##--- create the initialization method
    def __init__(self, cfd_position_records, inverse_position_records):
        self.net_actual_position_records    = cfd_position_records.copy()
        self.gross_actual_position_records  = cfd_position_records.copy()
        self.net_inverse_position_records   = inverse_position_records.copy()
        self.gross_inverse_position_records = inverse_position_records.copy()
        self.net_actual_periapsis_list      = []
        self.gross_actual_periapsis_list    = []
        self.net_inverse_periapsis_list     = []
        self.gross_inverse_periapsis_list   = []
        self.net_actual_apoapsis_list       = []
        self.gross_actual_apoapsis_list     = []
        self.net_inverse_apoapsis_list      = []
        self.gross_inverse_apoapsis_list    = []
        self.net_actual_evasive_list        = []
        self.gross_actual_evasive_list      = []
        self.net_inverse_evasive_list       = []
        self.gross_inverse_evasive_list     = []
        
    #+----------------------------------------------------------------------------+
    #| @func: PnL Model                                                           |
    #| @desc: analysis tool for plotting PnL against the actual and inverse data  |
    #| @params: N/A                                                               |
    #| @return: N/A                                                               |
    #+----------------------------------------------------------------------------+
    def pnl_model(self, net_actual_position_records, gross_actual_position_records, net_inverse_position_records, gross_inverse_position_records):
        n = min(len(net_actual_position_records), len(gross_actual_position_records), len(net_inverse_position_records), len(gross_inverse_position_records))
        net_actual_position_records    = net_actual_position_records.iloc[:n]
        gross_actual_position_records  = gross_actual_position_records.iloc[:n]
        net_inverse_position_records   = net_inverse_position_records.iloc[:n]
        gross_inverse_position_records = gross_inverse_position_records.iloc[:n]
        trades = np.arange(1, n+1)

        actual_cum_net    = net_actual_position_records['Net Profit'].cumsum().values
        inverse_cum_net   = net_inverse_position_records['Net Profit'].cumsum().values
        actual_cum_gross  = gross_actual_position_records['Gross Profit'].cumsum().values
        inverse_cum_gross = gross_inverse_position_records['Gross Profit'].cumsum().values
        
        # --- Fit regressions
        actual_net_res, actual_net_best       = self.fit_multiple_regressions(trades, actual_cum_net)
        actual_gross_res, actual_gross_best   = self.fit_multiple_regressions(trades, actual_cum_gross)
        inverse_net_res, inverse_net_best     = self.fit_multiple_regressions(trades, inverse_cum_net)
        inverse_gross_res, inverse_gross_best = self.fit_multiple_regressions(trades, inverse_cum_gross)
        
        ##--- function for getting regression growth rates
        def regression_growth_rate(prediction, trades):
                return (prediction[-1] - prediction[0]) / (trades[-1] - trades[0])
        
        ##--- get regression growth rate
        actual_net_growth    = regression_growth_rate(actual_net_res[actual_net_best]['prediction'], trades)
        inverse_net_growth   = regression_growth_rate(inverse_net_res[inverse_net_best]['prediction'], trades)
        actual_gross_growth  = regression_growth_rate(actual_gross_res[actual_gross_best]['prediction'], trades)
        inverse_gross_growth = regression_growth_rate(inverse_gross_res[inverse_gross_best]['prediction'], trades)
        
        plt.style.use('dark_background')
        fig, axes = plt.subplots(2, 1, figsize=(16,8), sharex=True)
        
        def _plot_with_regression(ax, y_actual, y_inverse, actual_res, inverse_res, actual_best, inverse_best, actual_growth, inverse_growth, 
                                  ticks_actual, ticks_inverse, apo_actual, apo_inverse, evasive_actual, evasive_inverse,
                                  title, color_actual='deepskyblue', color_inverse='orange', alpha_fill=0.2):
            ##--- raw PnL
            ax.plot(trades, y_actual, lw=2.5, color=color_actual, label='Actual')
            ax.plot(trades, y_inverse, lw=2.5, linestyle='--', color=color_inverse, label='Inverse')
            
            # --- max divergence markers
            pnl_diff = y_actual - y_inverse
            max_div_idx = np.argmax(np.abs(pnl_diff))

            ax.axvline(trades[max_div_idx], color='white', linestyle=':', alpha=0.6)
            ax.scatter(trades[max_div_idx], y_actual[max_div_idx],
                    s=120, color=color_actual, edgecolors='black')
            ax.scatter(trades[max_div_idx], y_inverse[max_div_idx],
                    s=120, color=color_inverse, edgecolors='black')
            
            # --- Best-fit regression overlay
            ax.plot(trades, actual_res[actual_best]['prediction'], lw=2.5, color='lime', linestyle=':', label=f'Actual {actual_best.title()} Fit')
            ax.plot(trades, inverse_res[inverse_best]['prediction'], lw=2.5, color='red', linestyle=':', label=f'Inverse {inverse_best.title()} Fit')

            # --- Filled area under curves
            ax.fill_between(trades, y_actual, 0, where=y_actual>=0, color=color_actual, alpha=alpha_fill)
            ax.fill_between(trades, y_actual, 0, where=y_actual<0, color=color_actual, alpha=alpha_fill*0.7)
            ax.fill_between(trades, y_inverse, 0, where=y_inverse>=0, color=color_inverse, alpha=alpha_fill)
            ax.fill_between(trades, y_inverse, 0, where=y_inverse<0, color=color_inverse, alpha=alpha_fill*0.7)
            
            # zero line
            ax.axhline(0, color='white', linestyle='--', linewidth=1, alpha=0.6)
            
            # mark periapsis points
            for tick in ticks_actual:
                if tick <= len(trades):
                    ax.axvline(trades[tick - 1], color=color_actual, linestyle=':', alpha=0.6)
            for tick in ticks_inverse:
                if tick <= len(trades):
                    ax.axvline(trades[tick - 1], color=color_inverse, linestyle=':', alpha=0.6)

            # mark apoapsis points
            for tick in apo_actual:
                if tick <= len(trades):
                    ax.axvline(trades[tick - 1], color='lime', linestyle=':', alpha=0.6)
            for tick in apo_inverse:
                if tick <= len(trades):
                    ax.axvline(trades[tick - 1], color='red', linestyle=':', alpha=0.6)
                    
            # mark evasive maneuver points
            for tick in evasive_actual:
                if tick <= len(trades):
                    ax.axvline(trades[tick - 1], color='magenta', linestyle=':', alpha=0.6)
            for tick in evasive_inverse:
                if tick <= len(trades):
                    ax.axvline(trades[tick - 1], color='gold', linestyle=':', alpha=0.6)
            
            # --- regime labeling with minimum separation
            for y, color in zip([y_actual, y_inverse], [color_actual, color_inverse]):
                min_distance = int(0.05 * len(y))
                last_labeled_idx = -min_distance
                sign = np.sign(y)
                sign[sign == 0] = 1
                regime_start = 0
                for i in range(1, len(y)):
                    if sign[i] != sign[i-1]:
                        seg = slice(regime_start, i)
                        if sign[i-1] > 0:
                            idx = seg.start + np.argmax(y[seg])
                            va = 'bottom'
                        else:
                            idx = seg.start + np.argmin(y[seg])
                            va = 'top'
                        if idx - last_labeled_idx >= min_distance:
                            ax.plot(trades[idx], y[idx], 'o', color=color)
                            ax.text(trades[idx], y[idx],
                                    f"-${abs(y[idx]):.2f}" if y[idx] < 0 else f"${y[idx]:.2f}",
                                    color=color, fontsize=9, ha='left', va=va, fontweight='bold')
                            last_labeled_idx = idx
                        regime_start = i
                # final regime
                seg = slice(regime_start, len(y))
                if sign[-1] > 0:
                    idx = seg.start + np.argmax(y[seg])
                    va = 'bottom'
                else:
                    idx = seg.start + np.argmin(y[seg])
                    va = 'top'
                if idx - last_labeled_idx >= min_distance:
                    ax.plot(trades[idx], y[idx], 'o', color=color)
                    ax.text(trades[idx], y[idx], f"{y[idx]:.2f}", color=color,
                            fontsize=9, ha='left', va=va, fontweight='bold')

            # --- move Max ΔPnL label to bottom-left
            ax.text(
                0.01, 1.02,
                f"Max ΔPnL = {pnl_diff[max_div_idx]:.2f}",
                transform=ax.transAxes,
                fontsize=10,
                ha='left',
                va='bottom',
                fontweight='bold',
                bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=3)
            )
            
            growth_ratio = actual_growth / inverse_growth if inverse_growth != 0 else np.nan
            ax.text(
                0.11, 1.02,
                (
                    f" |  Actual ΔPnL / ΔTrades:  ${actual_growth:.4f}\n"
                    f" |  Inverse ΔPnL / ΔTrades: ${inverse_growth:.4f}\n"
                    f" |  Ratio (A / I): {growth_ratio:.2f}"
                ),
                transform=ax.transAxes,
                fontsize=10,
                ha='left',
                va='bottom',
                fontweight='bold',
                bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', pad=4)
            )

            ax.axhline(0, color='white', linestyle='--', linewidth=1, alpha=0.6)
            ax.set_title(title, fontsize=16)
            ax.set_xlabel('Trade Count', fontsize=12)
            ax.set_ylabel('Cumulative PnL', fontsize=12)
            ax.grid(True, linestyle=':', alpha=0.4)
            # ax.legend(loc='lower left')
            
        ##--- plot Nt Profit PnL
        _plot_with_regression(axes[0], actual_cum_net, inverse_cum_net,
                            actual_net_res, inverse_net_res,
                            actual_net_best, inverse_net_best,
                            actual_net_growth, inverse_net_growth,
                            ticks_actual = self.net_actual_periapsis_list,
                            ticks_inverse = self.net_inverse_periapsis_list,
                            apo_actual = self.net_actual_apoapsis_list,
                            apo_inverse = self.net_inverse_apoapsis_list,
                            evasive_actual = self.net_actual_evasive_list,
                            evasive_inverse = self.net_inverse_evasive_list,
                            title = 'Cumulative Net Profit: Actual vs Inverse')

        ##--- plot Gross Profit PnL
        _plot_with_regression(axes[1], actual_cum_gross, inverse_cum_gross,
                            actual_gross_res, inverse_gross_res,
                            actual_gross_best, inverse_gross_best,
                            actual_gross_growth, inverse_gross_growth,
                            ticks_actual = self.gross_actual_periapsis_list,
                            ticks_inverse = self.gross_inverse_periapsis_list,
                            apo_actual = self.gross_actual_apoapsis_list,
                            apo_inverse = self.gross_inverse_apoapsis_list,
                            evasive_actual = self.gross_actual_evasive_list,
                            evasive_inverse = self.gross_inverse_evasive_list,
                            title = 'Cumulative Gross Profit: Actual vs Inverse')
        
        ##--- show the PnL chart
        plt.tight_layout()
        plt.show()

    #+----------------------------------------------------------------------------+
    #| @func: Fit Multiple Regressions                                            |
    #| @desc: finds the regression of best fit for the position records data      |
    #| @params: N/A                                                               |
    #| @return: N/A                                                               |
    #+----------------------------------------------------------------------------+
    def fit_multiple_regressions(self, trades, pnl_values):
            results = {}
            X = trades.reshape(-1, 1)

            # Linear
            lin = LinearRegression().fit(X, pnl_values)
            lin_pred = lin.predict(X)
            results['linear'] = {
                'prediction': lin_pred,
                'model': lin,
                'r2': r2_score(pnl_values, lin_pred)
            }

            # Quadratic
            poly = PolynomialFeatures(degree=2)
            X_quad = poly.fit_transform(X)
            quad = LinearRegression().fit(X_quad, pnl_values)
            quad_pred = quad.predict(X_quad)
            results['quadratic'] = {'prediction': quad_pred, 'model': quad, 'poly': poly,
                                    'r2': r2_score(pnl_values, quad_pred)}

            # Logarithmic
            X_log = np.log(X + 1e-6)
            log_reg = LinearRegression().fit(X_log, pnl_values)
            log_pred = log_reg.predict(X_log)
            results['logarithmic'] = {
                'prediction': log_pred,
                'model': log_reg,
                'r2': r2_score(pnl_values, log_pred)
            }

            # Exponential
            mask = pnl_values > 0
            if np.any(mask):
                X_exp = X[mask]
                y_exp = np.log(pnl_values[mask])
                exp_reg = LinearRegression().fit(X_exp, y_exp)
                exp_pred = np.exp(exp_reg.predict(X))
                results['exponential'] = {
                    'prediction': exp_pred,
                    'model': exp_reg,
                    'r2': r2_score(pnl_values, exp_pred)
                }
            else:
                results['exponential'] = {
                    'prediction': np.zeros_like(pnl_values),
                    'model': None,
                    'r2': -np.inf
                }

            ##--- return best fit regression
            best = max(results.items(), key=lambda kv: kv[1]['r2'])[0]
            return results, best

    #+----------------------------------------------------------------------------+
    #| @func: Print Regression Equation                                           |
    #| @desc: prints the regression equation for fit multiple regressions to test |
    #| @params: N/A                                                               |
    #| @return: N/A                                                               |
    #+----------------------------------------------------------------------------+
    def print_regression_equation(self, res, best, label=''):
        print(f"\n===== {label} REGRESSION MODEL =====")
        print(f"Best model: {best}")

        model = res[best]['model']

        if best == 'linear':
            a = model.coef_[0]
            b = model.intercept_
            print(f"y = {a:.6f} * x + {b:.6f}")

        elif best == 'quadratic':
            # sklearn PolynomialFeatures order: [1, x, x^2]
            c = model.intercept_
            b = model.coef_[1]
            a = model.coef_[2]
            print(f"y = {a:.6f} * x^2 + {b:.6f} * x + {c:.6f}")

        elif best == 'logarithmic':
            a = model.coef_[0]
            b = model.intercept_
            print(f"y = {a:.6f} * ln(x) + {b:.6f}")

        elif best == 'exponential':
            # your exp model is: y = exp(a*x + b)
            a = model.coef_[0]
            b = model.intercept_
            print(f"y = exp({a:.6f} * x + {b:.6f})")

        print("===================================")
    
    #+----------------------------------------------------------------------------+
    #| @func: Compute Periapsis                                                   |
    #| @desc: finds the point of highest velocity in the position records data    |
    #| @params: N/A                                                               |
    #| @return: N/A                                                               |
    #+----------------------------------------------------------------------------+
    def compute_periapsis(self, res, best, trades):
        x = trades.astype(float)
        i = x[-1]

        velocities = None
        candidate_periapsis = None

        ##---------- LINEAR ----------
        if best == 'linear':
            model = res['linear']['model']
            a = model.coef_[0]
            b = model.intercept_

            y_reg = a * x + b
            velocities = np.full_like(x, a)
            
            ##--- model regression in polar space
            theta_reg = np.arctan2(y_reg, x)     # angle for regression
            r_reg     = np.sqrt(x**2 + y_reg**2) # radius for regression

            ##--- model first derivative in polar space
            theta_vel = np.arctan2(velocities, x)     # angle for derivative
            r_vel     = np.sqrt(x**2 + velocities**2) # radius for derivative

            ##--- find intersection between regresssion and first derivative in polar space
            theta_diff          = np.abs(theta_reg - theta_vel) # difference between regression and first derivative angles
            intersection_idx    = np.argmin(theta_diff)         # intersection point of regression and first derivative
            candidate_periapsis = int(x[intersection_idx])      # x-value of intersection: periapsis
        
        ##---------- QUADRATIC ----------
        elif best == 'quadratic':
            model = res['quadratic']['model']
            poly = res['quadratic']['poly']

            a = model.coef_[2]
            b = model.coef_[1]
            c = model.intercept_

            X_poly = poly.transform(x.reshape(-1, 1))
            y_reg = model.predict(X_poly)
            velocities = 2 * a * x + b
            
            ##--- model regression in polar space
            theta_reg = np.arctan2(y_reg, x)     # angle for regression
            r_reg     = np.sqrt(x**2 + y_reg**2) # radius for regression

            ##--- model first derivative in polar space
            theta_vel = np.arctan2(velocities, x)     # angle for derivative
            r_vel     = np.sqrt(x**2 + velocities**2) # radius for derivative

            ##--- find intersection between regresssion and first derivative in polar space
            theta_diff          = np.abs(theta_reg - theta_vel) # difference between regression and first derivative angles
            intersection_idx    = np.argmin(theta_diff)         # intersection point of regression and first derivative
            candidate_periapsis = int(x[intersection_idx])      # x-value of intersection: periapsis
        
        ##---------- LOGARITHMIC ----------
        elif best == 'logarithmic':
            model = res['logarithmic']['model']
            a = model.coef_[0]
            b = model.intercept_

            y_reg = a * np.log(x) + b
            velocities = a / x
            
            ##--- model regression in polar space
            theta_reg = np.arctan2(y_reg, x)     # angle for regression
            r_reg     = np.sqrt(x**2 + y_reg**2) # radius for regression

            ##--- model first derivative in polar space
            theta_vel = np.arctan2(velocities, x)     # angle for derivative
            r_vel     = np.sqrt(x**2 + velocities**2) # radius for derivative

            ##--- find intersection between regresssion and first derivative in polar space
            theta_diff          = np.abs(theta_reg - theta_vel) # difference between regression and first derivative angles
            intersection_idx    = np.argmin(theta_diff)         # intersection point of regression and first derivative
            candidate_periapsis = int(x[intersection_idx])      # x-value of intersection: periapsis
        
        ##---------- EXPONENTIAL ----------
        elif best == 'exponential':
            model = res['exponential']['model']
            a = model.coef_[0]
            b = model.intercept_

            y_reg = np.exp(a * x + b)
            velocities = a * y_reg
            
            ##--- model regression in polar space
            theta_reg = np.arctan2(y_reg, x)     # angle for regression
            r_reg     = np.sqrt(x**2 + y_reg**2) # radius for regression

            ##--- model first derivative in polar space
            theta_vel = np.arctan2(velocities, x)     # angle for derivative
            r_vel     = np.sqrt(x**2 + velocities**2) # radius for derivative

            ##--- find intersection between regresssion and first derivative in polar space
            theta_diff          = np.abs(theta_reg - theta_vel) # difference between regression and first derivative angles
            intersection_idx    = np.argmin(theta_diff)         # intersection point of regression and first derivative
            candidate_periapsis = int(x[intersection_idx])      # x-value of intersection: periapsis
        else:
            return None

        ##--- return the periapsis
        periapsis = candidate_periapsis
        return periapsis

    #+----------------------------------------------------------------------------+
    #| @func: Compute Apoapsis                                                    |
    #| @desc: finds the point of lowest velocity using inverse polar intersection |
    #| @params: N/A                                                               |
    #| @return: apoapsis x-value or None                                          |
    #+----------------------------------------------------------------------------+
    def compute_apoapsis(self, res, best, trades):
        x = trades.astype(float)

        candidate_apoapsis = None

        # ---------- LINEAR ----------
        if best == 'linear':
            # Linear inverse is still linear → no curvature → reject
            return None

        # ---------- QUADRATIC ----------
        elif best == 'quadratic':
            model = res['quadratic']['model']
            poly  = res['quadratic']['poly']

            a = model.coef_[2]
            b = model.coef_[1]
            c = model.intercept_

            # Forward regression
            X_poly = poly.transform(x.reshape(-1, 1))
            y_reg  = model.predict(X_poly)

            # Inverse derivative: dx/dy = 1 / (dy/dx)
            dy_dx = 2 * a * x + b
            dx_dy = np.where(dy_dx != 0, 1 / dy_dx, np.nan)

            # --- inverse regression in polar space
            theta_inv = np.arctan2(x, y_reg)
            r_inv     = np.sqrt(y_reg**2 + x**2)

            # --- inverse derivative in polar space
            theta_vel = np.arctan2(dx_dy, y_reg)
            r_vel     = np.sqrt(y_reg**2 + dx_dy**2)

            theta_diff = np.abs(theta_inv - theta_vel)
            intersection_idx = np.nanargmin(theta_diff)

            candidate_apoapsis = int(x[intersection_idx])

        # ---------- LOGARITHMIC ----------
        elif best == 'logarithmic':
            model = res['logarithmic']['model']
            a = model.coef_[0]
            b = model.intercept_

            y_reg = a * np.log(x) + b
            dy_dx = a / x
            dx_dy = np.where(dy_dx != 0, 1 / dy_dx, np.nan)

            theta_inv = np.arctan2(x, y_reg)
            r_inv     = np.sqrt(y_reg**2 + x**2)

            theta_vel = np.arctan2(dx_dy, y_reg)
            r_vel     = np.sqrt(y_reg**2 + dx_dy**2)

            theta_diff = np.abs(theta_inv - theta_vel)
            intersection_idx = np.nanargmin(theta_diff)

            candidate_apoapsis = int(x[intersection_idx])

        # ---------- EXPONENTIAL ----------
        elif best == 'exponential':
            model = res['exponential']['model']
            a = model.coef_[0]
            b = model.intercept_

            y_reg = np.exp(a * x + b)
            dy_dx = a * y_reg
            dx_dy = np.where(dy_dx != 0, 1 / dy_dx, np.nan)

            theta_inv = np.arctan2(x, y_reg)
            r_inv     = np.sqrt(y_reg**2 + x**2)

            theta_vel = np.arctan2(dx_dy, y_reg)
            r_vel     = np.sqrt(y_reg**2 + dx_dy**2)

            theta_diff = np.abs(theta_inv - theta_vel)
            intersection_idx = np.nanargmin(theta_diff)

            candidate_apoapsis = int(x[intersection_idx])
        else:
            return None
        
        ##--- return the apoapsis
        apoapsis = candidate_apoapsis
        return apoapsis

    #+----------------------------------------------------------------------------+
    #| @func: Scan Orbit                                                          |
    #| @desc: determines if the trade PnL is next to apoapsis/periapsis extrema   |
    #| @params: N/A                                                               |
    #| @return: periapsis --> list of all periapses for the dataset               |
    #|           apoapsis --> list of all apoapses for the dataset                |
    #+----------------------------------------------------------------------------+
    def scan_orbit(self, trades, pnl, min_period = 3, max_window = 39):
        periapsis = []
        apoapsis  = []

        last_idx = 0
        state = "SEARCH_APOAPSIS"

        for i in range(min_period, len(trades)+1):
            # --- sliding window logic
            start_idx = max(last_idx, i - max_window)
            trades_subset = trades[start_idx:i]
            pnl_subset    = pnl[start_idx:i]

            if len(trades_subset) < min_period:
                continue

            res, best = self.fit_multiple_regressions(trades_subset, pnl_subset)

            if state == "SEARCH_PERIAPSIS":
                peri = self.compute_periapsis(res, best, trades_subset)
                if peri is not None and peri == len(trades_subset) - 1:
                    global_idx = start_idx + peri
                    periapsis.append(global_idx)
                    last_idx = global_idx
                    state = "SEARCH_APOAPSIS"

            elif state == "SEARCH_APOAPSIS":
                apo = self.compute_apoapsis(res, best, trades_subset)
                if apo is not None and apo == len(trades_subset) - 1:
                    global_idx = start_idx + apo
                    apoapsis.append(global_idx)
                    last_idx = global_idx
                    state = "SEARCH_PERIAPSIS"

        return periapsis, apoapsis

    #+----------------------------------------------------------------------------+
    #| @func: Detect Evasive Volume Reduction                                     |
    #| @desc: identifies trades where downside PnL velocity is abnormally large   |
    #| @params: pnl_series  -> cumulative or per-trade Net Profit (pd.Series)     |
    #|          window      -> rolling window for normalization                   |
    #|          z_thresh    -> downside z-score trigger                           |
    #|          confirm     -> consecutive confirmations required                 |
    #| @return: list of trade indices where volume should be reduced              |
    #+----------------------------------------------------------------------------+
    def detect_evasive_volume_reduction(self, pnl_series, window = 20, z_thresh = 2.5, confirm = 2):
        # --- per-trade velocity
        v = pnl_series.diff()

        # --- rolling MAD (robust volatility of velocity)
        rolling_mad = (
            v.rolling(window)
            .apply(lambda x: np.median(np.abs(x - np.median(x))), raw=True)
        )

        # --- convert MAD to std-equivalent
        sigma_v = 1.4826 * rolling_mad

        # --- normalized velocity
        z_v = v / sigma_v

        evasive_idx = []
        hit_count = 0

        for i in range(len(z_v)):
            if z_v.iloc[i] < -z_thresh:
                hit_count += 1
            else:
                hit_count = 0

            if hit_count >= confirm:
                evasive_idx.append(i)
                hit_count = 0  # prevent repeated triggers

        return evasive_idx

    #+----------------------------------------------------------------------------+
    #| @func: Apply Volume Increase                                               |
    #| @desc: makes routine volume increases after prospective periapsis points   |
    #| @params: N/A                                                               |
    #| @return: N/A                                                               |
    #+----------------------------------------------------------------------------+
    def apply_volume_dilation(self, position_records, events, factor=1.00, window=14):
        ##--- sort event indices just in case
        events = sorted(events)

        n = len(position_records)

        for idx in events:
            start_idx = idx
            end_idx   = min(idx + window, n - 1)

            if start_idx >= n:
                continue

            position_records.loc[start_idx:end_idx, 'Volume'] *= factor
            position_records.loc[start_idx:end_idx, 'Gross Profit'] *= factor
            position_records.loc[start_idx:end_idx, 'Commission'] *= factor
            position_records.loc[start_idx:end_idx, 'Swap'] *= factor

            position_records.loc[start_idx:end_idx, 'Net Profit'] = (
                position_records.loc[start_idx:end_idx, 'Gross Profit'] +
                position_records.loc[start_idx:end_idx, 'Commission'] +
                position_records.loc[start_idx:end_idx, 'Swap']
            )
            
            position_records.loc[start_idx:end_idx, 'Volume'] = (
                position_records.loc[start_idx:end_idx, 'Volume']
                .clip(lower=position_records['Volume'].median() * 0.25)
            )

        return position_records

    #+----------------------------------------------------------------------------+
    #| @func: Get All Periapsis Data                                              |
    #| @desc: perform analysis to obtain all the periapsis values                 |
    #| @params: N/A                                                               |
    #| @return: N/A                                                               |
    #+----------------------------------------------------------------------------+
    def get_all_periapsis_data(self):
        ##--- setup up data for periapsis analysis
        n = min(len(self.net_actual_position_records), len(self.net_inverse_position_records))
        self.net_actual_position_records  = (
            self.net_actual_position_records.iloc[:n]
            .reset_index(drop=True)
        )
        self.net_inverse_position_records = (
            self.net_inverse_position_records.iloc[:n]
            .reset_index(drop=True)
        )
        self.gross_actual_position_records = (
            self.gross_actual_position_records.iloc[:n]
            .reset_index(drop=True)
        )
        self.gross_inverse_position_records = (
            self.gross_inverse_position_records.iloc[:n]
            .reset_index(drop=True)
        )
        trades = np.arange(1, n + 1)
        
        ##--- get PnL
        actual_net_subset    = self.net_actual_position_records['Net Profit'].cumsum()
        actual_gross_subset  = self.gross_actual_position_records['Gross Profit'].cumsum()
        inverse_net_subset   = self.net_inverse_position_records['Net Profit'].cumsum()
        inverse_gross_subset = self.gross_inverse_position_records['Gross Profit'].cumsum()
        
        ##--- get apoapses and periapses
        self.net_actual_periapsis_list, self.net_actual_apoapsis_list       = self.scan_orbit(trades, actual_net_subset, min_period = 3, max_window = 39)
        self.gross_actual_periapsis_list, self.gross_actual_apoapsis_list   = self.scan_orbit(trades, actual_gross_subset, min_period = 3, max_window = 39)
        self.net_inverse_periapsis_list, self.net_inverse_apoapsis_list     = self.scan_orbit(trades, inverse_net_subset, min_period = 3, max_window = 48)
        self.gross_inverse_periapsis_list, self.gross_inverse_apoapsis_list = self.scan_orbit(trades, inverse_gross_subset, min_period = 3, max_window = 48)

        ##--- apply volume upgrades at periapses and apoapses
        self.net_actual_position_records    = self.apply_volume_dilation(self.net_actual_position_records, self.net_actual_apoapsis_list, factor = 2.00, window=14)
        self.net_actual_position_records    = self.apply_volume_dilation(self.net_actual_position_records, self.net_actual_periapsis_list, factor = 0.50, window=14)
        self.gross_actual_position_records  = self.apply_volume_dilation(self.gross_actual_position_records, self.gross_actual_apoapsis_list, factor = 2.00, window=14)
        self.gross_actual_position_records  = self.apply_volume_dilation(self.gross_actual_position_records, self.gross_actual_periapsis_list, factor = 0.50, window=14)
        self.net_inverse_position_records   = self.apply_volume_dilation(self.net_inverse_position_records, self.net_inverse_apoapsis_list, factor = 2.00, window=14)
        self.net_inverse_position_records   = self.apply_volume_dilation(self.net_inverse_position_records, self.net_inverse_periapsis_list, factor = 0.50, window=14)
        self.gross_inverse_position_records = self.apply_volume_dilation(self.gross_inverse_position_records, self.gross_inverse_apoapsis_list, factor = 2.00, window=14)
        self.gross_inverse_position_records = self.apply_volume_dilation(self.gross_inverse_position_records, self.gross_inverse_periapsis_list, factor = 0.50, window=14)

        ##--- redefine PnL
        actual_net_subset_2    = self.net_actual_position_records['Net Profit'].cumsum()
        actual_gross_subset_2  = self.gross_actual_position_records['Gross Profit'].cumsum()
        inverse_net_subset_2   = self.net_inverse_position_records['Net Profit'].cumsum()
        inverse_gross_subset_2 = self.gross_inverse_position_records['Gross Profit'].cumsum()
        
        ##--- detect evasive maneuvers
        self.net_actual_evasive_list    = self.detect_evasive_volume_reduction(actual_net_subset_2, window = 14, z_thresh = 2.0, confirm = 2)
        self.gross_actual_evasive_list  = self.detect_evasive_volume_reduction(actual_gross_subset_2, window = 14, z_thresh = 2.0, confirm = 2)
        self.net_inverse_evasive_list   = self.detect_evasive_volume_reduction(inverse_net_subset_2, window = 14, z_thresh = 2.0, confirm = 2)
        self.gross_inverse_evasive_list = self.detect_evasive_volume_reduction(inverse_gross_subset_2, window = 14, z_thresh = 2.0, confirm = 2)
        
        ##--- throttle volume at evasive maneuvers levels
        self.net_actual_position_records    = self.apply_volume_dilation(self.net_actual_position_records, self.net_actual_evasive_list, factor = 0.50, window=14)
        self.gross_actual_position_records  = self.apply_volume_dilation(self.gross_actual_position_records, self.gross_actual_evasive_list, factor = 0.50, window=14)
        self.net_inverse_position_records   = self.apply_volume_dilation(self.net_inverse_position_records, self.net_inverse_evasive_list, factor = 0.50, window=14)
        self.gross_inverse_position_records = self.apply_volume_dilation(self.gross_inverse_position_records, self.gross_inverse_evasive_list, factor = 0.50, window=14)
        
        self.pnl_model(self.net_actual_position_records, self.gross_actual_position_records, self.net_inverse_position_records, self.gross_inverse_position_records)


##--- execute the main method
if __name__ == '__main__':
    ##--- change directory to parent directory
    os.chdir(get_parent_directory())

    ##--- read the position data
    cfd_position_records     = pd.read_csv(r'data/complete_cfd_position_records.csv')
    inverse_position_records = pd.read_csv(r'data/inverse_cfd_position_records.csv')
    
    ##--- initialize the edge evaluator class
    periapsis_theory_modeler = Periapsis_Theory_Base_Modeler(cfd_position_records = cfd_position_records, inverse_position_records = inverse_position_records)
    
    ##--- perform edge analysis
    periapsis_theory_modeler.get_all_periapsis_data()