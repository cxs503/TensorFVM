"""Publication figures and paper-structured report from saved raw fields."""
import csv
import html
import json
from pathlib import Path
import textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from .core import metric_value

plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,
                     "axes.labelsize":11,"axes.titlesize":11,
                     "savefig.dpi":300,"pdf.fonttype":42,"ps.fonttype":42})


def figure(fig, directory, name):
    fig.tight_layout()
    for ext in ("png","pdf","svg"):
        fig.savefig(directory/f"{name}.{ext}",bbox_inches="tight")
    plt.close(fig)


def render_case(directory, row):
    fields=np.load(directory/"fields.npz",allow_pickle=False)
    out=directory/"figures";out.mkdir(exist_ok=True)
    x,y=fields["x_m"],fields["y_m"];X,Y=np.meshgrid(x,y)
    for key,label,cmap,name in [("p_pa","Pressure (Pa)","coolwarm","pressure"),
        ("u_m_s","u (m/s)","coolwarm","u"),
        ("v_m_s","v (m/s)","coolwarm","v")]:
        fig,ax=plt.subplots(figsize=(7,3 if row["case"]=="poiseuille" else 5))
        im=ax.pcolormesh(X,Y,fields[key],shading="nearest",cmap=cmap,rasterized=True)
        fig.colorbar(im,ax=ax,label=label);ax.set(xlabel="x (m)",ylabel="y (m)",
              title=f"{row['case']}: {row['resolution']}")
        ax.set_aspect("equal")
        figure(fig,out,name)
    fig,ax=plt.subplots(figsize=(7,3 if row["case"]=="poiseuille" else 5))
    speed=np.hypot(fields["u_m_s"],fields["v_m_s"])
    im=ax.pcolormesh(X,Y,speed,shading="nearest",cmap="viridis",rasterized=True)
    fig.colorbar(im,ax=ax,label="Speed (m/s)");ax.set(xlabel="x (m)",ylabel="y (m)",title="Computed velocity magnitude")
    ax.set_aspect("equal");figure(fig,out,"speed")
    fig,ax=plt.subplots(figsize=(6,4))
    ax.plot(fields["pressure_x_m"],fields["pressure_curve_pa"],"o",ms=3,label="TensorLBM" if row["case"]=="taylor-green-lbm" else "FVM")
    ax.plot(fields["pressure_x_m"],fields["reference_pressure_curve_pa"],"-",lw=1.5,label="Analytic reference")
    ax.set(xlabel="x (m)",ylabel="Pressure (Pa)",title="Pressure comparison in declared region/time")
    ax.legend();ax.grid(alpha=.3);figure(fig,out,"pressure-curve")
    fig,ax=plt.subplots(figsize=(5,4))
    ax.plot(fields["profile_u_m_s"],fields["profile_y_m"],"o",ms=3,label="TensorLBM" if row["case"]=="taylor-green-lbm" else "FVM")
    ax.plot(fields["reference_u_profile_m_s"],fields["profile_y_m"],"-",label="Analytic")
    ax.set(xlabel="u (m/s)",ylabel="y (m)",title="Velocity profile at documented sample location")
    ax.legend();ax.grid(alpha=.3);figure(fig,out,"velocity-profile")
    history=json.loads((directory/"history.json").read_text())
    fig,ax=plt.subplots(figsize=(6,4))
    if row["case"]=="poiseuille":
        for key in ("continuity","momentum","mass_imbalance"):
            ax.semilogy(range(1,len(history)+1),[max(float(h[key]),1e-20) for h in history],label=key)
        ax.set(xlabel="SIMPLE iteration",ylabel="Residual",title="Convergence history")
    elif row["case"]=="taylor-green-lbm":
        t=[h["time_s"] for h in history]
        ax.plot(t,[h["kinetic_J"] for h in history],label="LBM computed KE")
        ax.set(xlabel="Time (s)",ylabel="Kinetic energy per m depth (J/m)",title="LBM energy diagnostic")
    else:
        t=[h["time_s"] for h in history]
        ax.plot(t,[h["kinetic_after_J"] for h in history],label="Computed KE")
        c=row["config"];volume=np.prod(np.array(c["lengths_m"])/[c["nx"],c["ny"],c["nz"]])
        initial=.5*c["density_kg_m3"]*volume*np.sum(fields["initial_faces_m_s"]**2)
        nu=c["viscosity_pa_s"]/c["density_kg_m3"]
        ax.plot(t,initial*np.exp(-4*nu*np.array(t)),"--",label="Continuous reference sampled on faces")
        ax.set(xlabel="Time (s)",ylabel="Kinetic energy (J)",title="Energy decay")
    ax.legend();ax.grid(alpha=.3);figure(fig,out,"history")
    # Numerical values underlying comparison figures are also portable CSV.
    for name,keys in [("pressure-curve",("pressure_x_m","pressure_curve_pa","reference_pressure_curve_pa")),
                      ("velocity-profile",("profile_y_m","profile_u_m_s","reference_u_profile_m_s"))]:
        with (directory/f"{name}.csv").open("w",newline="") as stream:
            writer=csv.writer(stream);writer.writerow(keys);writer.writerows(zip(*(fields[k] for k in keys)))


def render_study(output, summary):
    figs=output/"figures";figs.mkdir(exist_ok=True)
    for case in ("poiseuille","taylor-green"):
        rows=[r for r in summary["runs"] if r["case"]==case and r["role"]=="spatial"]
        fig,ax=plt.subplots(figsize=(6,4))
        for metric in ("velocity_l2","pressure_gradient" if case=="poiseuille" else "pressure_l2"):
            ax.loglog([r["spacing_m"] for r in rows],[100*metric_value(r,metric) for r in rows],"o-",label=metric)
        ax.axhline(3,color="k",ls="--",label="3% gate")
        ax.set(xlabel="Grid spacing (m)",ylabel="Relative error (%)",title=f"{case}: spatial refinement")
        ax.legend();ax.grid(which="both",alpha=.3);figure(fig,figs,f"{case}-spatial")
    rows=[r for r in summary["runs"] if r["role"]=="temporal"]
    fig,ax=plt.subplots(figsize=(6,4))
    ax.loglog([r["config"]["time_step_s"] for r in rows],
              [metric_value(r,"temporal_velocity_l2") for r in rows],"o-",label="Fixed-grid exact exponential")
    ax.set(xlabel="Time step (s)",ylabel="Relative temporal error",title="Taylor-Green: temporal refinement")
    ax.legend();ax.grid(which="both",alpha=.3);figure(fig,figs,"taylor-green-temporal")


def write_report(output, summary):
    """A printable HTML, English manuscript PDF, Chinese report and editable TeX."""
    rows=summary["runs"];regular=[r for r in rows if r["role"]!="negative-control"]
    controls=[r for r in rows if r["role"]=="negative-control"]
    tables=[]
    for r in rows:
        pressure="pressure_gradient" if r["case"]=="poiseuille" else "pressure_l2"
        tables.append(f"| {r['case']} | {r['role']} | {r['resolution']} | "
          f"{r['config'].get('time_step_s','—')} | {100*metric_value(r,'velocity_l2'):.6f} | "
          f"{100*metric_value(r,pressure):.6f} | {'PASS' if r['passed'] else 'FAIL'} |")
    finest={case:[r for r in rows if r["case"]==case and r["role"]=="spatial"][-1] for case in ("poiseuille","taylor-green")}
    result_table="\n".join(["| Case | Study | Grid | dt (s) | Velocity L2 (%) | Pressure metric (%) | All gates |",
                            "|---|---|---|---|---:|---:|---|",*tables])
    text=f"""# TensorFVM 不可压有限体积方法的解析基准验证
## 摘要
本研究通过共性 benchmark 模块，对平行板 Poiseuille 流和二维 Taylor–Green 衰减涡开展空间及时间加密。所有正式算例的预声明速度、压力与壁面相关误差均要求严格小于 3%，并分别满足质量、求解残差和适用的能量门。总体验收：**{'通过' if summary['passed'] else '失败'}**。失败对照单独列出并保留原始数据。本文验证解析子问题，不能外推三维湍流、复杂曲面或上浮破冰。

## 1. 问题介绍与数学模型
常密度不可压方程为 ∇·u=0，ρ(∂u/∂t+∇·(u⊗u))=−∇p+∇·[μ(∇u+∇uᵀ)]。本研究无体力。全部单位为 SI，二维力/流量按单位展向深度计算。

### 1.1 Poiseuille 通道
L=0.12 m，H=0.02 m，ρ=1000 kg/m³，ν=10⁻⁵ m²/s，Ub=5×10⁻⁴ m/s，Re_H=1。入口规定均匀速度，壁面无滑移，出口零表压。均匀入口存在发展段；解析验证只在 3H≤x≤5H 的压力/壁面窗口，以及 x 最接近 5H 的速度剖面进行。全域云图展示实际解，入口区没有被当作充分发展解析解。
解析解：u(y)=6Ub(y/H)(1−y/H)，dp/dx=−12μUb/H²，|τw|=6μUb/H。压力曲线只在比较窗内对齐均值，不把任意表压零点计入误差。

### 1.2 Taylor–Green 衰减涡
周期域 [0,2π]³ m，ρ=1 kg/m³，μ=0.1 Pa·s，U0=1 m/s，波数 k=1 m⁻¹。二维解析涡嵌入四层展向三维网格，w=0；不是三维湍流。
u=sin(x)cos(y)e^(−2νt)，v=−cos(x)sin(y)e^(−2νt)，p=ρ[cos(2x)+cos(2y)]e^(−4νt)/4。
速度比较位于实际错位面、t=0.5 s。投影压力是末步 midpoint 压力，比较时间为 t−dt/2，压力采用零均值规范。云图中的中心速度仅用于展示，验收使用权威面速度。

## 2. 软件与共性模块
案例适配器提供配置、求解器、参考公式和原始场；共性 core 模块统一误差、验收、导出、来源绑定与观察阶；report 模块统一云图、曲线、报告；audit 模块独立读取原场重算指标。没有复制流体求解器，也没有把解析解作为计算结果输出。

安装：python -m pip install -e '.[benchmark,dev]'
运行：OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m tensorfvm.verification --output docs/verification-benchmarks
审计：python -m tensorfvm.verification.audit --directory docs/verification-benchmarks
失败返回非零退出码；--max-channel-iterations 1 可运行非收敛检查。输出目录须为空，防止覆盖既有失败证据。

## 3. 数值方法与实验设计
通道采用交错有限体积 SIMPLE、半单元无滑移黏性壁面离散，求解 tolerance=10⁻⁶，最多1000迭代。三网格36×12、72×24、144×48。空间误差以解析速度、压力梯度和压力曲线、壁面剪切及力平衡独立评估。
周期 MAC 采用权威面速度、共享中心动量通量、完整对称黏性应力、离散相容 FFT 投影和隐式 midpoint；非线性真实残差约10⁻¹² m/s。空间32²/48²/64²×4、dt=0.0125 s；时间32²×4、dt=0.1/0.05/0.025 s，全部至t=0.5 s。时间观察阶使用固定网格离散特征值的精确指数参考，避免混合空间误差。该参考不作为连续物理误差。
误差采用 ||q−q_ref||₂/||q_ref||₂ 与峰值归一化 L∞；无逐点除以参考零值。观察阶 p=ln(e₁/e₂)/ln(h₁/h₂)，来自实际解析误差，不使用假定阶或两网格 GCI 宣称验证完成。
正式算例全部需通过，失败对照不计入正例资格。非收敛对照与16²粗网格压力失败仅用于证明门能拒绝错误，绝不改成通过。

## 4. 计算结果
{result_table}

空间与时间观察阶：{json.dumps(summary['observed_orders'],ensure_ascii=False)}。
每个 result.json 包含全部门槛、指标、时间和计算配置；history.json/CSV 保存实际迭代/步历史；fields.npz 保存原始面场、中心展示场、参考与采样曲线，Taylor–Green 还保存全部接受步面场。

## 5. 压力场、速度场与剖面
周期涡压力曲线取中心平面 y=(floor(n/2)+1/2)Δy；速度剖面取u面 x=(floor(n/4)+1)Δx。通道剖面位置见每个reference记录。
"""
    for case,r in finest.items():
        d=r["directory"]
        text+=f"\n### {case}，{r['resolution']}\n"
        for name,caption in [("pressure","实际压力场，单位 Pa"),("speed","实际速度幅值，单位 m/s"),
            ("u","实际水平速度"),("v","实际垂向速度"),("pressure-curve","压力曲线与解析参考；使用上述区域、规范和时钟"),
            ("velocity-profile","速度剖面与解析参考"),("history","实际收敛或能量历史")]:
            text+=f"\n![{caption}]({d}/figures/{name}.png)\n\n图：{caption}。\n"
    for name in ("poiseuille-spatial","taylor-green-spatial","taylor-green-temporal"):
        text+=f"\n![{name}](figures/{name}.png)\n"
    text+=f"""
## 6. 讨论与适用边界
3% 是本项目预声明的解析验证门，并非论文、行业或工程安全的通用标准。多项解析误差与数值门均通过后，只认证本文的边界条件和评估区域。通道入口不是充分发展解；周期涡无固壁、自由液面和移动体。二维涡嵌入3D不能验证真实3D湍流。本研究不认证翼型/圆柱升阻力、SUBOFF、双向FSI或冰破坏。网格加密使用相同物理时间；未将小残差等同于物理误差。
另一次128²×4、dt=0.0125 s试验因Picard候选非有限被拒绝，完整失败场在相邻 verification-benchmarks-failed-n128 目录保留。它不计入正式32/48/64序列，不能被该序列通过覆盖。
历史根目录 tensorlbm 实验代码没有作为本报告求解后端；其投影尺度、裁剪和守恒问题仍须独立处理。此工作没有放宽旧失败门或覆盖旧证据。

## 7. 可复现性与数据可用性
源码 SHA256、git 基线、Python/Torch/NumPy、设备与线程见 summary.json；全部报告、图和数据摘要见 manifest.json。检查源码哈希与文件哈希后，再从原始场独立重算验收。图以300 dpi PNG、矢量PDF及SVG提供；压力/速度曲线同时导出CSV。报告PDF、可打印HTML及LaTeX稿可供论文编排；投稿前仍需期刊模板、作者信息和同行审查。
环境：{json.dumps(summary['provenance'],ensure_ascii=False)}。

## 8. 参考依据
解析参考由本文控制方程直接推导：通道积分 μu''=dp/dx 且满足无滑移与平均流量；周期涡代入不可压动量方程得到指数衰减及压力。时间参考来自离散相容拉普拉斯 Fourier 特征值。未采用无法追溯的经验升阻力值。

[1] G. I. Taylor and A. E. Green, Mechanism of the production of small eddies from large ones, Proceedings of the Royal Society A 158 (1937), 499–521. DOI: https://doi.org/10.1098/rspa.1937.0036 。本文使用其经典涡验证思想的二维解析衰减变体。
[2] NASA NPARC Alliance, Examining Spatial (Grid) Convergence, https://www.grc.nasa.gov/www/wind/valid/tutorial/spatconv.html 。用于网格研究方法背景，不将该网页当作本文3%阈值的来源。
"""
    if (output/"comparison.md").exists():
        text+="\n## 9. FVM 与 LBM 对照\n\n"+(output/"comparison.md").read_text()
    (output/"report.md").write_text(text)
    # Printable HTML uses escaped prose and all generated figures, without web dependencies.
    blocks=[]
    in_table=False
    for line in text.splitlines():
        if line.startswith("|"):
            if set(line.replace("|","").replace("-","").replace(":","").strip())==set():
                continue
            if not in_table:
                blocks.append("<table>");in_table=True
            blocks.append("<tr>"+"".join("<td>"+html.escape(cell.strip())+"</td>" for cell in line.strip("|").split("|"))+"</tr>")
            continue
        if in_table:
            blocks.append("</table>");in_table=False
        if line.startswith("!["):
            alt,src=line[2:].split("](",1)
            blocks.append(f'<figure><img src="{html.escape(src[:-1])}"><figcaption>{html.escape(alt)}</figcaption></figure>')
        elif line.startswith("#"):
            level=len(line)-len(line.lstrip("#"));blocks.append(f"<h{level}>{html.escape(line[level:].strip())}</h{level}>")
        elif line.strip():blocks.append("<p>"+html.escape(line)+"</p>")
    (output/"report.html").write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>TensorFVM verification</title><style>body{max-width:1000px;margin:40px auto;font:16px/1.6 sans-serif}img{max-width:100%}figure{break-inside:avoid}p{overflow-wrap:anywhere}table{border-collapse:collapse;font-size:13px}td{border:1px solid #777;padding:5px}@media print{body{margin:0}h2{break-before:page}}</style><body>'+"\n".join(blocks)+"</body></html>")
    paper_pdf(output,summary,finest)
    tex_report(output,summary,finest)


def paper_pdf(output,summary,finest):
    """English manuscript pages plus figure plates; no external TeX required."""
    chapters=[
     ("TensorFVM: analytic verification of incompressible finite-volume solvers",
      "Abstract\nTwo independent solver adapters share one benchmark pipeline. Plane Poiseuille flow tests a no-slip steady channel; a periodic Taylor-Green vortex tests transient face momentum. Every declared regular run must have velocity, pressure and applicable wall errors strictly below 3%, together with separate solver/mass/energy gates. Negative controls remain archived. Overall regular acceptance: "+str(summary["passed"])+"."),
     ("1. Mathematical model and boundary conditions",
      "Incompressible momentum: div(u)=0; rho(du/dt+div(u tensor u))=-grad(p)+div(mu(grad(u)+grad(u)^T)). All quantities are SI.\n\nChannel: L=0.12 m, H=0.02 m, rho=1000 kg/m^3, nu=1e-5 m^2/s, Ub=5e-4 m/s, Re_H=1. Uniform velocity inlet, no-slip walls and zero gauge-pressure outlet. Analytic comparison is confined to developed stations 3H<=x<=5H. Reference: u=6Ub(y/H)(1-y/H); dp/dx=-12mu Ub/H^2; |tau_w|=6mu Ub/H.\n\nPeriodic vortex: [0,2pi]^3 m; rho=1 kg/m^3, mu=0.1 Pa s, U0=1 m/s. u=sin(x)cos(y)exp(-2nu t); v=-cos(x)sin(y)exp(-2nu t); w=0. p=rho[cos(2x)+cos(2y)]exp(-4nu t)/4. It is a 2D exact solution embedded in four z layers, not 3D turbulence. Pressure is compared at the last midpoint t-dt/2, with zero mean."),
     ("2. Numerical methods and verification protocol",
      "Channel: staggered finite volumes and SIMPLE, half-cell wall diffusion, tolerance 1e-6. Grids 36x12, 72x24 and 144x48. Vortex: authoritative MAC face velocities, shared centered dual-volume momentum fluxes, complete symmetric stress, compatible periodic FFT pressure projection and implicit midpoint. Spatial grids 32x32x4, 48x48x4, 64x64x4 at dt=0.0125 s. Time steps 0.1, 0.05 and 0.025 s on 32x32x4. All vortex runs end at 0.5 s.\n\nRelative L2 uses global reference norm; relative Linf uses reference peak. Pressure gauge/time/locations are explicitly aligned. Temporal order uses a fixed-grid exact discrete eigenmode exponential; this is not a continuous physical reference. Observed order is computed from actual errors and refinement ratios. No assumed-order GCI is presented. Negative controls: unconverged channel and coarse vortex pressure failure."),
     ("3. Results and discussion",
      "Regular-run acceptance: "+str(summary["passed"])+".\nObserved orders: "+json.dumps(summary["observed_orders"])+"\n\nThe following table reports actual computed errors. All detailed numerical gates, configurations and clocks are in result.json. Fields.npz stores authoritative and visualization fields; full accepted vortex face states and iteration histories are retained. Figures use actual computed values, physical coordinates and labelled SI color bars."),
     ("4. Scope, reproducibility and conclusions",
      "The 3% gate is a declared project analytic-verification target, not a universal publication or safety standard. Channel entrance flow is excluded from developed-flow comparisons. Vortex periodic accuracy does not establish solid-wall, free-surface, moving-body, high-Re turbulence, airfoil or SUBOFF accuracy. Repository tensorlbm experiments are not used as these solver backends.\n\nInstall: python -m pip install -e '.[benchmark,dev]'\nRun: python -m tensorfvm.verification --output docs/verification-benchmarks\nAudit: python -m tensorfvm.verification.audit --directory docs/verification-benchmarks\n\nSource hashes, git base, environment and artifact hashes are supplied. PNG at 300 dpi and PDF/SVG figures, CSV curves, raw NPZ, full JSON histories and editable TeX accompany this manuscript. The analytic references are derived directly from the governing equations; discrete time references use the compatible Fourier eigenvalue.\n\nA separate 128x128x4 run with dt=0.0125 was rejected by the Picard nonlinear solver at its third step, after 0.025 s. Nonfinite candidate fields and unchanged accepted state are retained. The qualified 32/48/64 study does not certify that 128 grid.\n\nReferences: Taylor and Green, Proc. R. Soc. A 158 (1937), 499-521, DOI 10.1098/rspa.1937.0036 (classical vortex-verification background; this report uses a 2D decay variant). NASA NPARC, Examining Spatial (Grid) Convergence, grc.nasa.gov/www/wind/valid/tutorial/spatconv.html (methodology, not the source of a 3% standard).\n\nThis verification report requires author details, journal formatting and peer review before submission.")
    ]
    comparisons=[r for r in summary["runs"] if r["role"]=="comparison"]
    if comparisons:
        results="\n".join(f"LBM {r['resolution']}, Ma={r['config']['actual_reference_mach']:.4f}: u L2={100*metric_value(r,'velocity_l2'):.5f}%, p L2={100*metric_value(r,'pressure_l2'):.5f}%, passed={r['passed']}" for r in comparisons)
        chapters.append(("5. Matched TensorLBM comparison",
            "Same continuum periodic domain, rho, viscosity, velocity amplitude and end time. Analytic errors use each solver's actual DOF locations and correct pressure clock. LBM uses external production D2Q9 equilibrium/BGK/stream, without mass correction. Float64 populations retain the existing upstream float32 weight constants.\n\n"+results+"\n\nFVM advantages supported here are the incompressible pressure constraint, authoritative face continuity, discrete energy/work balance and no lattice Mach requirement on its physical time step. Primary fields per physical plane occupy 4N2 doubles versus D2Q9 9N2 doubles; this is not peak memory.\n\nObserved CPU timings include setup, diagnostics and recording, with different z replication and time steps. No FVM speed or peak-memory superiority is claimed. LBM local updates avoid a global pressure solve. This is one BGK baseline, not all TensorLBM capabilities."))
    with PdfPages(output/"report.pdf") as pdf:
        for title,body in chapters:
            fig=plt.figure(figsize=(8.27,11.69));fig.text(.09,.93,textwrap.fill(title,62),fontsize=15,va="top")
            wrapped="\n".join(textwrap.fill(p,88) if p else "" for p in body.splitlines())
            fig.text(.09,.86,wrapped,fontsize=10,va="top",linespacing=1.6)
            if title.startswith("3."):
                cells=[]
                for r in summary["runs"]:
                    pressure="pressure_gradient" if r["case"]=="poiseuille" else "pressure_l2"
                    setting=(f"Ma={r['config']['actual_reference_mach']:.3f}" if r["role"]=="comparison" else
                             f"dt={r['config']['time_step_s']:.4g}" if "time_step_s" in r["config"] else "--")
                    cells.append([r["case"],r["role"],r["resolution"],setting,f"{100*metric_value(r,'velocity_l2'):.5f}",f"{100*metric_value(r,pressure):.5f}","PASS" if r["passed"] else "FAIL"])
                ax=fig.add_axes([.07,.12,.86,.40]);ax.axis("off")
                table=ax.table(cellText=cells,colLabels=["Case","Study","Grid","dt / Ma","u L2 (%)","p (%)","Gate"],loc="center")
                table.auto_set_font_size(False);table.set_fontsize(7);table.scale(1,1.6)
            pdf.savefig(fig);plt.close(fig)
        for case,r in finest.items():
            for name in ("pressure","speed","u","v","pressure-curve","velocity-profile","history"):
                fig,ax=plt.subplots(figsize=(8.27,8));ax.imshow(plt.imread(output/r["directory"]/"figures"/(name+".png")));ax.axis("off")
                ax.set_title(f"{case}, {r['resolution']}: {name}\nPressure and velocity have documented different clocks where applicable.")
                pdf.savefig(fig,bbox_inches="tight");plt.close(fig)
        for name in ("poiseuille-spatial","taylor-green-spatial","taylor-green-temporal") + (("fvm-lbm-errors",) if comparisons else ()):
            fig,ax=plt.subplots(figsize=(8.27,8));ax.imshow(plt.imread(output/"figures"/(name+".png")));ax.axis("off")
            pdf.savefig(fig,bbox_inches="tight");plt.close(fig)


def tex_report(output,summary,finest):
    text=r"""\documentclass[11pt]{article}
\usepackage[a4paper,margin=25mm]{geometry}
\usepackage{amsmath,graphicx,booktabs,hyperref}
\title{Analytic verification of TensorFVM incompressible finite-volume solvers}
\author{Author details to be supplied}
\date{}
\begin{document}\maketitle
\begin{abstract}
A shared benchmark workflow evaluates plane Poiseuille flow and a periodic Taylor--Green vortex using raw numerical fields, declared relative error gates below 3 percent, refinement studies and independent replay. Negative controls are retained. These results establish bounded analytic verification, not engineering icebreaking qualification.
\end{abstract}
\section{Problem formulation}
For constant density, $\nabla\cdot\mathbf u=0$ and
$\rho(\partial_t\mathbf u+\nabla\cdot(\mathbf u\otimes\mathbf u))=-\nabla p+\nabla\cdot[\mu(\nabla\mathbf u+\nabla\mathbf u^T)]$.
The channel has $L=0.12$ m, $H=0.02$ m, $\rho=1000$ kg/m$^3$, $\nu=10^{-5}$ m$^2$/s and $U_b=5\times10^{-4}$ m/s. Uniform inlet and no-slip walls produce a developing entrance. Analytic comparison is restricted to $3H\leq x\leq5H$:
$u=6U_b(y/H)(1-y/H)$, $dp/dx=-12\mu U_b/H^2$, $|\tau_w|=6\mu U_b/H$.
Pressure curves are mean aligned only in this window.
The vortex has $[0,2\pi]^3$ m periodic boundaries, $\rho=1$, $\mu=0.1$, and
$u=\sin x\cos y e^{-2\nu t}$, $v=-\cos x\sin y e^{-2\nu t}$, $w=0$,
$p=\rho(\cos2x+\cos2y)e^{-4\nu t}/4$.
Velocity is sampled at its actual faces at $t=0.5$ s; pressure at cell centers and the last midpoint $t-\Delta t/2$, with zero mean.
\section{Methods and reproducibility}
Channel grids are $36\times12$, $72\times24$, $144\times48$, using staggered SIMPLE with $10^{-6}$ residual tolerance.
Periodic MAC uses compatible FFT projection, full symmetric stress and implicit midpoint. Spatial grids are $32^2$, $48^2$, $64^2$ with four z layers and $\Delta t=0.0125$ s. Temporal steps are $0.1$, $0.05$, $0.025$ s on $32^2\times4$.
Relative errors use global L2 or reference-peak Linf normalization. Time-order analysis uses the exact fixed-grid discrete eigenmode decay to isolate temporal error. Detailed metrics, clocks, source hashes, environment and raw fields accompany this manuscript.
\section{Results}
\resizebox{\linewidth}{!}{\begin{tabular}{llllrrl}\toprule
Case & Study & Grid & $\Delta t$ & $u$ L2 (\%) & $p$ metric (\%) & Gate\\\midrule
"""
    for r in summary["runs"]:
        pressure="pressure_gradient" if r["case"]=="poiseuille" else "pressure_l2"
        text+=f"{r['case']} & {r['role']} & {r['resolution']} & {format(r['config']['time_step_s'],'.6g') if 'time_step_s' in r['config'] else '--'} & {100*metric_value(r,'velocity_l2'):.5f} & {100*metric_value(r,pressure):.5f} & {'PASS' if r['passed'] else 'FAIL'} "+chr(92)*2+"\n"
    text+=r"\bottomrule\end{tabular}}"+"\n"
    for case,r in finest.items():
        for name in ("pressure","speed","pressure-curve","velocity-profile","history"):
            text+=r"\begin{figure}[htbp]\centering\includegraphics[width=0.9\linewidth]{"+r["directory"]+"/figures/"+name+".pdf}"+r"\caption{"+case+": "+name+r". Actual computed fields, SI units; comparison uses the documented reference region, gauge and time.}\end{figure}"+"\n"
    comparisons=[r for r in summary["runs"] if r["role"]=="comparison"]
    if comparisons:
        text+=r"\section{Matched TensorLBM comparison}"+"\n"
        text+=r"Both solvers use the same periodic continuum problem, SI parameters and end time. Errors use actual DOF locations and their respective pressure clocks. TensorLBM calls unmodified external D2Q9 BGK modules. The FVM advantages established here are incompressible pressure/face continuity, a discrete energy balance and time stepping independent of a lattice Mach mapping. Primary state storage per physical plane is four versus nine double arrays, not a peak-memory measurement. Different dimensions, step counts and diagnostic overhead prevent speed-superiority claims."+"\n"
        for row in comparisons:
            text+=f"LBM {row['resolution']}: velocity L2 {100*metric_value(row,'velocity_l2'):.5f} percent, pressure L2 {100*metric_value(row,'pressure_l2'):.5f} percent. "+"\n"
        text+=r"\begin{figure}[htbp]\centering\includegraphics[width=\linewidth]{figures/fvm-lbm-errors.pdf}\caption{Matched analytic accuracy with declared Mach sensitivity.}\end{figure}"+"\n"
    text+=r"""\section{Limitations and conclusion}
The vortex is a two-dimensional exact solution embedded in a three-dimensional mesh, not a turbulence benchmark. No qualification is claimed for free surfaces, moving bodies, ice fracture or coupled SUBOFF cases. The 3 percent criterion is a project target, not a universal publication standard. The analytic references follow directly by substitution into the governing equations. Negative controls are archived and excluded explicitly from regular-run qualification. A separate 128-square-grid run at dt=0.0125 failed its nonlinear solve; the retained failure must not be overwritten by the qualified 32/48/64 study. See the companion report and machine-readable summary for observed orders and all gate results.
\begin{thebibliography}{9}
\bibitem{tg} G. I. Taylor and A. E. Green, Mechanism of the production of small eddies from large ones, Proc. R. Soc. A 158 (1937), 499--521. DOI: 10.1098/rspa.1937.0036.
\bibitem{nasa} NASA NPARC Alliance, Examining Spatial (Grid) Convergence, \url{https://www.grc.nasa.gov/www/wind/valid/tutorial/spatconv.html}.
\end{thebibliography}
\end{document}
"""
    (output/"report.tex").write_text(text)


def comparison_report(output,summary):
    """Evidence-based strengths, with matched physics and explicit cost limitations."""
    rows=[r for r in summary["runs"] if r["role"]=="comparison"]
    fvm=[r for r in summary["runs"] if r["case"]=="taylor-green" and r["role"]=="spatial"]
    figures=output/"figures"
    matched=[]
    for lbm in rows:
        n=lbm["config"]["nx"];f=next(r for r in fvm if r["config"]["nx"]==n)
        lf=np.load(output/lbm["directory"]/"fields.npz");ff=np.load(output/f["directory"]/"fields.npz")
        # Slice a single plane to avoid crediting replication in z as extra physical resolution.
        fbytes=(3*n*n+n*n)*8
        lbytes=9*n*n*8
        matched.append(dict(grid=n,mach=lbm["config"]["actual_reference_mach"],
          fvm_velocity_error=metric_value(f,"velocity_l2"),lbm_velocity_error=metric_value(lbm,"velocity_l2"),
          fvm_pressure_error=metric_value(f,"pressure_l2"),lbm_pressure_error=metric_value(lbm,"pressure_l2"),
          fvm_authoritative_divergence=max(h["max_divergence_s_inv"] for h in json.loads((output/f["directory"]/"history.json").read_text())),
          lbm_central_divergence=float(np.max(abs(lf["divergence_s_inv"]))),
          lbm_mass_drift=max(h["relative_mass_drift"] for h in json.loads((output/lbm["directory"]/"history.json").read_text())),
          fvm_primary_bytes_per_plane=fbytes,lbm_primary_bytes_per_plane=lbytes,
          fvm_elapsed_s=f["elapsed_s"],lbm_elapsed_s=lbm["elapsed_s"],
          fvm_steps=f["iterations"],lbm_steps=lbm["iterations"],
          fvm_pressure_time_s=f["pressure_time_s"],lbm_pressure_time_s=lbm["pressure_time_s"]))
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    labels=[f"N={r['grid']}, Ma={r['mach']:.3f}" for r in matched];xx=np.arange(len(matched))
    for ax,key in zip(axes,("velocity","pressure")):
        ax.bar(xx-.18,[100*r[f"fvm_{key}_error"] for r in matched],width=.36,label="FVM MAC")
        ax.bar(xx+.18,[100*r[f"lbm_{key}_error"] for r in matched],width=.36,label="TensorLBM BGK")
        ax.axhline(3,color="k",ls="--",label="3% gate");ax.set_xticks(xx,labels,rotation=15)
        ax.set(ylabel="Relative L2 error (%)",title=key+" accuracy");ax.legend()
    figure(fig,figures,"fvm-lbm-errors")
    table=["| N | LBM Ma | FVM u L2 (%) | LBM u L2 (%) | FVM p L2 (%) | LBM p L2 (%) | FVM / LBM steps | FVM / LBM observed CPU seconds |",
           "|---:|---:|---:|---:|---:|---:|---|---|"]
    for r in matched:
        table.append(f"| {r['grid']} | {r['mach']:.5f} | {100*r['fvm_velocity_error']:.5f} | {100*r['lbm_velocity_error']:.5f} | {100*r['fvm_pressure_error']:.5f} | {100*r['lbm_pressure_error']:.5f} | {r['fvm_steps']} / {r['lbm_steps']} | {r['fvm_elapsed_s']:.4f} / {r['lbm_elapsed_s']:.4f} |")
    text="""# FVM 与 TensorLBM 的匹配解析对照

## 1. 公平性与共同条件
两者求解同一周期二维 Taylor–Green 连续问题：L=2π m、ρ=1 kg/m³、ν=0.1 m²/s、U0=1 m/s、终止时间0.5 s、CPU单线程、float64状态。同样的L2/峰值归一化L∞定义与严格3%精度门；解析值在各自实际自由度位置计算。
FVM 是四层展向的MAC原型，物理解无展向变化；LBM是二维D2Q9 BGK，调用外部仓库原始 equilibrium、collide_bgk、stream、macroscopic 共性模块，没有在benchmark复制碰撞/迁移算法。LBM根据dx/dt映射SI，并使整数步恰好到0.5 s；τ=0.5+3νdt/dx²，Ma按实际dt记录。初始密度由解析压力和EOS确定，populations采用生产equilibrium初始化，可能产生有限启动误差；未做人为质量调平。
FVM压力属于最后midpoint时刻；LBM压力属于步末。各自对照对应时刻的解析压力，没有把两个不同时钟的压力直接相减。LBM节点和FVM错位面不同，比较解析误差而非未经插值的数组差。
LBM使用float64 populations，但当前生产D2Q9权重由float32常数形成；本报告保留该实现并测量质量漂移，不把它归因于所有LBM方法。对照仅涉及此BGK基线，不能代表TensorLBM的MRT、cumulant或GPU优化能力。

## 2. 实际结果
"""
    text+="\n".join(table)+"\n\n![Accuracy comparison](figures/fvm-lbm-errors.png)\n"
    text+="""
## 3. FVM 的优势及证据边界
- 压力由不可压约束求解，不依靠弱可压状态方程或人为声速；本例记录相容面散度与实际压力误差。
- MAC 面速度直接满足离散连续性，并保留每步动量、压力功、黏性功与能量账本。LBM的中心差分散度是可压宏观诊断，与MAC权威通量散度不是同一个离散算子；只并列展示，不据此判定LBM错误。
- 物理时间步不由格子声速映射决定。本例FVM可以直接做0.1/0.05/0.025 s时间加密；隐式非线性求解仍可能失败，绝不宣称任意大步长。
- 按一层物理平面统计，FVM主未知量3速度+1压力为4N²个double，D2Q9为9N²个double，比例4/9。此数只描述主状态存储，不含FVM非线性临时量、FFT、LBM流索引、报告缓存；不是峰值内存或总成本优势。
- 新通道报告另验证压力梯度、壁面剪切与力平衡，给后续FEM牵引映射提供明确SI量。不能据此声称已有曲面、移动体或工程FSI优势。

## 4. 性能与LBM自身优势
此处运行时间来自单次CPU完整执行，包含初始化、诊断与原场采样；FVM实际推进四个z平面，LBM只有一个二维网格，而且时间步数不同。未进行重复统计、暖启动、峰值RSS或GPU公平性能测量。因此不宣称FVM速度更快或峰值内存更低。LBM的局部碰撞/迁移、无需全局压力Poisson，以及成熟融合GPU路径具有不同成本结构，应在后续相同精度、相同物理维度的基准中实测。
两者的失败、质量漂移和压力误差均保留，不能通过筛选指标来制造优势。未来应增加相同固壁边界的压力驱动通道和同一几何外流，在完整3%门与守恒约束下比较达到指定精度的总成本。

## 5. 数据与复现
summary.json各LBM运行记录外部仓库提交号、共性源码SHA、Mach、τ、dt、密度变化与质量历史。原始populations及每一步完整状态可独立重建BGK和迁移。comparison.json绑定此次测得指标。主报告的FVM资格与LBM比较资格分别记录；某一基线失败不能被另一个基线的成功覆盖。
"""
    (output/"comparison.md").write_text(text)
    from .core import write_json
    write_json(output/"comparison.json",dict(schema="tensor-suite.periodic-comparison/1",
       matched=matched,accuracy_limit=.03,lbm_accuracy_passed=summary["comparison_passed"],
       performance_superiority_claimed=False,comparison_scope="matched analytic physics; different DOFs and pressure clocks"))
    with PdfPages(output/"comparison.pdf") as pdf:
        fig=plt.figure(figsize=(8.27,11.69));fig.text(.08,.94,"FVM / TensorLBM matched Taylor-Green comparison",fontsize=15)
        body=("Same L=2pi m, rho=1 kg/m3, nu=0.1 m2/s, U0=1 m/s, final time 0.5 s, CPU single thread, float64 states. "
              "FVM uses actual MAC face locations and four replicated z planes; LBM uses D2Q9 node locations. "
              "Pressure errors use the correct individual midpoint/end clocks.\n\n"
              "FVM strengths: incompressible pressure constraint, discrete face continuity, stepwise momentum/work/energy audit, physical time step independent of lattice Mach scaling. "
              "Primary state storage per physical plane is 4N2 doubles versus 9N2 D2Q9 populations. This is not a peak-memory result.\n\n"
              "LBM is run with unmodified production equilibrium, BGK collision and stream modules. Upstream weights retain float32 constants in float64 populations. "
              "Mach sensitivity and population histories are retained. This compares one BGK baseline, not all TensorLBM capabilities.\n\n"
              "Observed single-run CPU times include diagnostics and sampling. Different z replication, time steps and solver setup prevent a speed superiority claim. "
              "LBM local updates avoid global pressure solves. No GPU or peak RSS claim is made.")
        fig.text(.08,.88,"\n".join(textwrap.fill(p,90) for p in body.splitlines()),va="top",fontsize=10,linespacing=1.6)
        cells=[[str(r["grid"]),f"{r['mach']:.4f}",f"{100*r['fvm_velocity_error']:.4f}",f"{100*r['lbm_velocity_error']:.4f}",f"{100*r['fvm_pressure_error']:.4f}",f"{100*r['lbm_pressure_error']:.4f}"] for r in matched]
        ax=fig.add_axes([.06,.14,.88,.22]);ax.axis("off");t=ax.table(cellText=cells,colLabels=["N","LBM Ma","FVM u %","LBM u %","FVM p %","LBM p %"],loc="center");t.auto_set_font_size(False);t.set_fontsize(9);t.scale(1,1.5)
        pdf.savefig(fig);plt.close(fig)
        fig,ax=plt.subplots(figsize=(10,5));ax.imshow(plt.imread(figures/"fvm-lbm-errors.png"));ax.axis("off");pdf.savefig(fig,bbox_inches="tight");plt.close(fig)


def regenerate(directory):
    """Rebuild publications from frozen raw results; permit only report-code changes."""
    from .core import sha,write_json,artifact_manifest
    from .audit import audit
    directory=Path(directory);summary=json.loads((directory/"summary.json").read_text())
    root=Path(__file__).resolve().parents[3]
    report_name="src/tensorfvm/verification/report.py"
    previous=summary["provenance"]["source_sha256"][report_name]
    for name,digest in summary["provenance"]["source_sha256"].items():
        if name!=report_name and sha(root/name)!=digest:
            raise ValueError("numerical/driver source changed; rerun benchmark instead: "+name)
    # Validate prior evidence before changing any report file.
    manifest=json.loads((directory/"manifest.json").read_text())
    for name,digest in manifest["artifacts_sha256"].items():
        if sha(directory/name)!=digest:raise ValueError("raw/report artifact changed: "+name)
    summary["provenance"]["source_sha256"][report_name]=sha(Path(__file__))
    summary["publication_regenerated_from_frozen_fields"]=True
    write_json(directory/"summary.json",summary)
    for row in summary["runs"]:render_case(directory/row["directory"],row)
    render_study(directory,summary)
    if any(r["role"]=="comparison" for r in summary["runs"]):comparison_report(directory,summary)
    write_report(directory,summary)
    # The old artifact hashes necessarily refer to the old publication.
    (directory/"manifest.json").unlink()
    write_json(directory/"audit.json",audit(directory,source_check=True))
    write_json(directory/"publication-generation.json",dict(
       previous_report_source_sha256=previous,report_source_sha256=sha(Path(__file__)),
       numerical_and_driver_sources_unchanged=True,
       raw_fields_unchanged=True,scope="report regenerated; numerical steps independently re-audited"))
    write_json(directory/"manifest.json",dict(schema="tensorfvm.benchmark-manifest/1",
       source_sha256=summary["provenance"]["source_sha256"],artifacts_sha256=artifact_manifest(directory)))


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description="Regenerate paper reports from frozen benchmark fields")
    parser.add_argument("--directory",type=Path,required=True)
    args=parser.parse_args()
    regenerate(args.directory)
