#!/usr/bin/env python3
"""Simple core functionality test without numpy dependency."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_basic_functionality():
    """Test basic functionality without numpy."""
    try:
        # Test basic imports
        from analog_pde_solver.core.solver import AnalogPDESolver
        from analog_pde_solver.core.equations import PoissonEquation
        print("✅ Core imports successful")
        
        # Create simple solver instance (this will work with our fallback)
        solver = AnalogPDESolver(crossbar_size=32)
        print(f"✅ Solver created: {type(solver)}")
        
        # Create PDE equation
        pde = PoissonEquation(domain_size=(32,))
        print(f"✅ PDE created: {type(pde)}")
        
        # Test RTL generation (no numpy needed)
        from analog_pde_solver.rtl.verilog_generator import VerilogGenerator
        rtl_gen = VerilogGenerator()
        verilog_code = rtl_gen.generate_top_module(16, 1, "poisson")
        print(f"✅ RTL generated: {len(verilog_code)} characters")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ General error: {e}")
        return False


if __name__ == "__main__":
    success = test_basic_functionality()
    print(f"\nTest result: {'✅ PASSED' if success else '❌ FAILED'}")
    sys.exit(0 if success else 1)
