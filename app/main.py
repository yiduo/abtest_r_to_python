from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import numpy as np
from scipy import stats
import math
import os

app = FastAPI()

# 获取当前文件所在目录的路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 挂载静态文件和模板
# app.mount(
#     "/static", StaticFiles(directory=os.path.join(current_dir, "static")), name="static"
# )
templates = Jinja2Templates(directory=os.path.join(current_dir, "templates"))


def try_ie(func):
    """错误处理装饰器"""

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            return "错误: 请仔细检查输入的参数是否合理。"

    return wrapper


@try_ie
def calculate_prop_test(avg_rr, lift, sig_level, power_level=0.8):
    """计算比例检验所需样本量"""
    p1 = avg_rr / 100
    p2 = p1 * (1 + lift / 100)  # 修改这里，因为前端传入的lift是百分比

    # 使用正态分布近似计算样本量
    z_alpha = abs(stats.norm.ppf((1 - sig_level) / 2))
    z_beta = abs(stats.norm.ppf(1 - power_level))

    p_bar = (p1 + p2) / 2

    n = math.ceil(
        (
            z_alpha * np.sqrt(2 * p_bar * (1 - p_bar))
            + z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
        )
        ** 2
        / (p1 - p2) ** 2
    )
    return n


@try_ie
def calculate_t_test(delta, sd, sig_level, power_level=0.8):
    """计算t检验所需样本量，使用与 R power.t.test 相同的计算方法"""
    try:
        # 转换所有输入为 np.float64，确保最高精度
        delta = np.float64(delta)
        sd = np.float64(sd)
        alpha = np.float64(1 - sig_level)  # 转换显著性水平为 alpha
        power_level = np.float64(power_level)

        print("\n=== Python 计算过程 ===")
        print("输入参数:")
        print(f"delta = {delta}")
        print(f"sd = {sd}")
        print(f"alpha = {alpha}")
        print(f"power = {power_level}")

        def power_t_test_iter(n):
            # 计算非中心参数
            lambda_val = (delta / sd) * np.sqrt(n / 2)
            # 自由度
            df = 2 * (n - 1)
            # 计算 t 值
            t_crit = abs(stats.t.ppf(alpha / 2, df))
            # 计算检验力
            power = (
                1
                - stats.nct.cdf(t_crit, df, lambda_val)
                + stats.nct.cdf(-t_crit, df, lambda_val)
            )
            return power - power_level

        # 使用正态分布进行初始估计
        z_alpha = abs(stats.norm.ppf(alpha / 2))
        z_beta = abs(stats.norm.ppf(1 - power_level))
        n_initial = 2 * ((sd / delta) * (z_alpha + z_beta)) ** 2

        print("\n初始估计:")
        print(f"z_alpha = {z_alpha:.16f}")
        print(f"z_beta = {z_beta:.16f}")
        print(f"n_initial = {n_initial:.16f}")

        # 使用二分法查找精确的样本量
        n_low = 2
        n_high = max(100, math.ceil(n_initial * 2))

        while n_high - n_low > 0.01:
            n_mid = (n_low + n_high) / 2
            if power_t_test_iter(n_mid) < 0:
                n_low = n_mid
            else:
                n_high = n_mid

        n_exact = n_high
        n = math.ceil(n_exact)

        print("\n最终结果:")
        print(f"n_exact = {n_exact:.16f}")
        print(f"n (向上取整) = {n}")

        return max(2, n)

    except Exception as e:
        print(f"错误: {str(e)}")
        raise ValueError(str(e))


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/calculate/prop")
async def calculate_proportion(
    request: Request,
    avg_rr: float = Form(...),
    lift: float = Form(...),
    sig_level: float = Form(...),
    num_groups: int = Form(...),
    daily_traffic: int = Form(None),
    time_period: int = Form(None),
):
    n = calculate_prop_test(avg_rr, lift, float(sig_level))
    total_sample = n * num_groups

    result = {
        "per_group": n,
        "total": total_sample,
    }

    if daily_traffic is not None:
        result["time_needed"] = math.ceil(total_sample / daily_traffic)

    if time_period is not None:
        result["daily_needed"] = math.ceil(total_sample / time_period)

    return result


@app.post("/calculate/mean")
async def calculate_mean(
    request: Request,
    delta: float = Form(...),  # 最小可检测差异（绝对值）
    sd: float = Form(...),  # 标准差
    sig_level: float = Form(...),  # 显著性水平
    num_groups: int = Form(...),  # 总组数
    daily_traffic: int = Form(None),
    time_period: int = Form(None),
):
    try:
        print("\n=== 开始计算均值检验样本量 ===")
        # 确保参数为正确的数值类型
        delta = abs(float(delta))  # 确保差值为正数
        sd = float(sd)
        sig_level = float(sig_level)
        num_groups = int(num_groups)

        # 参数验证
        if delta <= 0:
            raise ValueError("最小可检测差异必须大于0")
        if sd <= 0:
            raise ValueError("标准差必须大于0")
        if sig_level <= 0 or sig_level >= 1:
            raise ValueError("显著性水平必须在0和1之间")
        if num_groups < 2:
            raise ValueError("组数必须大于等于2")

        # 计算样本量
        n = calculate_t_test(delta, sd, sig_level)
        total_sample = n * num_groups

        print(f"每组样本量: {n}")
        print(f"总样本量: {total_sample}")
        print("=== 计算完成 ===\n")

        result = {
            "per_group": n,
            "total": total_sample,
        }

        if daily_traffic is not None and daily_traffic > 0:
            result["time_needed"] = math.ceil(total_sample / daily_traffic)

        if time_period is not None and time_period > 0:
            result["daily_needed"] = math.ceil(total_sample / time_period)

        return result

    except ValueError as e:
        return {"error": str(e)}
    except Exception:
        return {"error": "错误: 请仔细检查输入的参数是否合理。"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
