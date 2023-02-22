from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(
    name="ngme_cpp",
    ext_modules=[
        cpp_extension.CppExtension(name="ngme_cpp", sources=["ngme.cpp"]),
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)

