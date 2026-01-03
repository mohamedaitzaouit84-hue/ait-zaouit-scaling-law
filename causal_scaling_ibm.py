from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit import QuantumCircuit, transpile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ---------- الاتصال ----------
API_KEY = "ضع_API_الخاص_بك_هنا"
service = QiskitRuntimeService(
    channel="ibm_quantum_platform",
    token=API_KEY,
    instance="open-instance"
)

backend = service.backend("ibm_torino")
sampler = Sampler(mode=backend)

network_sizes = [5, 10, 20, 40]
results = []

print("🔹 تشغيل الاختبارات على IBM Torino")

for N in network_sizes:
    print(f"▶ N = {N}")

    qc = QuantumCircuit(N)
    qc.h(range(N))
    qc.measure_all()

    isa_circuit = transpile(qc, backend=backend, optimization_level=1)

    job = sampler.run([isa_circuit], shots=1024)
    result = job.result()[0]
    counts = result.data.meas.get_counts()

    mean_ones = np.mean([s.count("1") for s in counts.keys()])
    std_ones  = np.std([s.count("1") for s in counts.keys()])

    results.append({"N": N, "Mean": mean_ones, "Std": std_ones})

    plt.figure(figsize=(6,3))
    plt.bar(counts.keys(), counts.values())
    plt.title(f"IBM Torino | N={N}")
    plt.xticks(rotation=90, fontsize=6)
    plt.tight_layout()
    plt.show()

df = pd.DataFrame(results)
display(df)

# تحليل اللاخطية
x = np.log(df["N"])
y = np.log(df["Std"])
beta, _ = np.polyfit(x, y, 1)
r2 = np.corrcoef(x, y)[0,1]**2

print(f"📐 Torino scaling: beta={beta:.3f}, R²={r2:.3f}")

plt.loglog(df["N"], df["Std"], "o-")
plt.xlabel("N")
plt.ylabel("Std(C)")
plt.title("Causal Asymmetry Scaling – IBM Torino")
plt.show()
