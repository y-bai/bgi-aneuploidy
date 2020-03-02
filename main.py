#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
    File Name: main.py
    Description:
    
Created by YongBai on 2020/3/2 2:22 PM.
"""

import logging
from utils import get_pos

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    re_df = get_pos(chr='13')
    logger.info(re_df.head())