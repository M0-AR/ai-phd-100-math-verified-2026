"""pytest: one test per chapter (10 tests cover 100 exercises with hard numeric asserts)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]/"src"))
from ai_phd_100 import (verify_ch01, verify_ch02, verify_ch03, verify_ch04, verify_ch05,
                        verify_ch06, verify_ch07, verify_ch08, verify_ch09, verify_ch10)
def test_ch01_shapes(): assert verify_ch01()["ex01_ops"] == 67108864
def test_ch02_spectral(): assert abs(verify_ch02()["ex20_cond"]-100.0) < 1e-9
def test_ch03_calculus(): assert verify_ch03()["ex23"] == 24
def test_ch04_backprop(): assert verify_ch04()["ex31_dw"] == 12
def test_ch05_probability(): assert abs(verify_ch05()["ex41"]-1/6) < 1e-9
def test_ch06_statistics(): assert abs(verify_ch06()["ex57_se"]-0.02) < 1e-12
def test_ch07_information(): assert abs(verify_ch07()["ex63"]-7.389056) < 0.001
def test_ch08_optimization(): assert abs(verify_ch08()["ex74"]-10.0) < 1e-9
def test_ch09_numerics(): assert verify_ch09()["ex90_bytes"] == 2**31
def test_ch10_transformer(): assert verify_ch10()["ex93_flops"] == 549755813888
