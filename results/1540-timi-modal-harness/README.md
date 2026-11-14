# Modal Vulkan probe

Zjistí, jestli na Modal L40S běží Vulkan s NVIDIA GPU (nutné pro Isaac Sim rendering).

Příkaz (z kořene projektu, po `modal setup` provedeném Timem):

    <HJ_ROOT>/.venv-modal/bin/modal run modal/vulkan_probe.py

Cena: < 0,50 USD (jedna L40S, timeout 600 s).

## Jak číst výsledek
- `vulkaninfo --summary` uvádí NVIDIA GPU (L40S) jako Vulkan device a v `icd.d` je `nvidia_icd.json` = GO.
- Jen `llvmpipe` (softwarový renderer) nebo chyba typu "failed to find a Vulkan ICD" / chybějící libnvidia-gl = NO-GO pro Isaac Sim.
