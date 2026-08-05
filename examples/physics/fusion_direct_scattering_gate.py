from __future__ import annotations

from reality_stone.clarus.fusion_direct_scattering_loop import (
    audit_direct_nuclear_scattering,
)


def main() -> None:
    audit = audit_direct_nuclear_scattering()
    print("CE FUSION DIRECT-OPERATOR NUCLEAR CONTROL")
    print(f" g_N                              {audit.required_nucleon_coupling:.9e}")
    print(f" scalar range fm                  {audit.scalar_range_fm:.9e}")
    print(f" Born scattering shift fm        {audit.free_born_scattering_length_shift_fm:.9e}")
    print(f" shift / triplet uncertainty     {audit.born_shift_to_triplet_uncertainty:.9e}")
    print(f" shift / singlet uncertainty     {audit.born_shift_to_singlet_uncertainty:.9e}")
    print(f" Hulthen deuteron shift keV      {audit.absolute_deuteron_shift_kev:.9e}")
    print(f" full nuclear refit              {audit.strong_potential_refit_performed}")
    print(f" experimental exclusion          {audit.experimental_exclusion_derived}")
    print(f" physical direct-operator gate   {audit.physical_direct_operator_gate_pass}")


if __name__ == "__main__":
    main()
