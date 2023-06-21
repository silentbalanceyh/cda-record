#!/usr/bin/env bash
# Jieba分词
conda install -c conda-forge jieba
# jieba_fast
# pip install jieba_fast
# XGBoost
conda install xgboost
# CatBoost
conda install catboost
# LightGBM
conda install lightgbm


# LightGBM
# brew install cmake libomp
# pip install lightgbm
# 问题：Library not loaded: /usr/local/opt/libomp/lib/libomp.dylib
# ln -s /opt/homebrew/opt/libomp/lib /usr/local/opt/libomp/lib