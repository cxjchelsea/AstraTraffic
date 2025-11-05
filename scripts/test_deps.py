# -*- coding: utf-8 -*-
"""
依赖检查工具
用于检查关键依赖包的安装情况
"""
import pkg_resources as P

def v(name):
    try:
        return P.get_distribution(name).version
    except:
        return 'NOT INSTALLED'

if __name__ == "__main__":
    print("langchain       =", v("langchain"))
    print("langchain-core  =", v("langchain-core"))
    print("langchain-comm. =", v("langchain-community"))
    print("langsmith       =", v("langsmith"))
    print("langchain-ollama=", v("langchain-ollama"))

