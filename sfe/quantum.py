import numpy as np


class QuantumCorrection:
    def __init__(self, T2: float, epsilon_0: float):
        self.T2 = float(T2)
        self.epsilon_0 = float(epsilon_0)

    def apply_lindblad_step(self, rho: np.ndarray, dt: float, scale: float = 1.0) -> np.ndarray:
        if self.T2 <= 0.0 or dt <= 0.0:
            return rho
        gamma = scale / (2.0 * self.T2)
        sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
        drho = gamma * (sigma_z @ rho @ sigma_z.conj().T - rho)
        return rho + dt * drho


class SFEFieldModel:
    def __init__(self, epsilon_0: float, T2: float):
        self.epsilon_0 = float(epsilon_0)
        self.T2 = float(T2) if T2 != 0.0 else 1.0
        self.omega = 2.0 * np.pi / self.T2

    def get_field_value(self, t: float) -> float:
        return self.epsilon_0 * float(np.cos(self.omega * t))


class SFEActiveCanceller:
    def __init__(self, qc: QuantumCorrection, efficiency: float):
        self.qc = qc
        self.efficiency = float(efficiency)
        self.sfe_model = SFEFieldModel(qc.epsilon_0, qc.T2)

    def simulate_with_active_cancellation(
        self,
        rho_init: np.ndarray,
        total_time: float,
        n_steps: int,
    ):
        dt = float(total_time) / float(n_steps) if n_steps > 0 else 0.0
        times = np.linspace(0.0, total_time - dt, n_steps) if n_steps > 0 else np.array([])
        rho = np.array(rho_init, dtype=complex)
        target = np.array(rho_init, dtype=complex)
        fids = []
        for t in times:
            rho = self.qc.apply_lindblad_step(rho, dt, scale=1.0)
            field = self.sfe_model.get_field_value(float(t))
            residual = (1.0 - self.efficiency) * field
            theta = residual * dt
            phase = np.exp(-0.5j * theta)
            U = np.array([[phase, 0.0], [0.0, np.conj(phase)]], dtype=complex)
            rho = U @ rho @ U.conj().T
            fids.append(_fidelity(rho, target))
        return times, np.array(fids)


def _fidelity(rho: np.ndarray, target: np.ndarray) -> float:
    return float(np.real(np.trace(target @ rho)))


