import logging
import os
import time

from redis import StrictRedis


def get_rc(func):
    def wrapper(self, *args, **kwargs):
        self._get_redis_conn()
        res = func(self, *args, **kwargs)
        return res

    return wrapper


class RedisClient(object):

    def __init__(self, host_and_ports, passwd, mode="cluster", max_connection_pool_flush_time=1800):
        self.host_and_ports = host_and_ports
        self.passwd = passwd
        self.mode = mode
        self._rc = None
        self.connection_pool_flush_time = -1
        self.max_connection_pool_flush_time = max_connection_pool_flush_time

    def _get_redis_conn_standalone(self):
        if self._rc is None:
            host_and_port_array = self.host_and_ports.strip().split(";")
            host = host_and_port_array[0].split(":")[0]
            port = int(host_and_port_array[0].split(":")[1])
            rc = StrictRedis(host=host, port=port, password=self.passwd)
            self._rc = rc
            self.connection_pool_flush_time = time.time()
        if time.time() - self.connection_pool_flush_time > self.max_connection_pool_flush_time:
            self._rc.connection_pool.disconnect()
            self.connection_pool_flush_time = time.time()

    def _get_redis_conn(self):
        self._get_redis_conn_standalone()

    @get_rc
    def setnx(self, key, value):
        return self._rc.setnx(key, value)

    @get_rc
    def get(self, key):
        return self._rc.get(key)

    @get_rc
    def expire(self, key, time_out=24 * 3600):
        return self._rc.expire(key, time_out)

    @get_rc
    def delete(self, key):
        return self._rc.delete(key)


if "k_bj_redis_host" not in os.environ:
    # for debug
    a = '''   k_bj_redis_host=10.16.2.92
        k_bj_redis_port=16379
        k_bj_redis_pwd=2ZH6MsUUVOo@'''
    for x in a.split("\n"):
        k, v = x.strip().split("=")
        os.environ[k] = v

KSYUN_REDIS_HOST = os.environ['k_bj_redis_host']
KSYUN_REDIS_PWD = os.environ['k_bj_redis_pwd']
KSYUN_REDIS_PORT = os.environ['k_bj_redis_port']

KSYUN_REDIS = RedisClient(f"{KSYUN_REDIS_HOST}:{KSYUN_REDIS_PORT}", KSYUN_REDIS_PWD, mode="standalone")


def lock_source_file(source_file, time_out=24 * 3600):
    key = f"DP_DATASET_TASK_{source_file}"
    set_res = KSYUN_REDIS.setnx(key, "RUNNING")
    if int(set_res) == 1:
        KSYUN_REDIS.expire(key, time_out)
        return 1
    else:
        return 0


def unlock_source_file(source_file):
    key = f"DP_DATASET_TASK_{source_file}"
    KSYUN_REDIS.delete(key)

