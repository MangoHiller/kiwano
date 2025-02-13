import os
from hostlist import expand_hostlist
import socket

def find_free_port():
    """Trouve un port TCP libre sur la machine"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))  # Laisse le système choisir un port libre
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]  # Récupère le numéro du port
    
if __name__ == "__main__":

    print(f"FREEEE :{find_free_port()}")
    
    # Récupération des variables SLURM
    slurm_nodelist = os.environ.get("SLURM_JOB_NODELIST", "non défini")
    slurm_nodeid = os.environ.get("SLURM_NODEID", "non défini")
    slurm_num_nodes = os.environ.get("SLURM_JOB_NUM_NODES", "non défini")
    slurm_procid = os.environ.get("SLURM_PROCID", "non défini")
    slurm_port = os.environ.get("MASTER_PORT", "29500")

    # Expansion des nœuds avec hostlist
    expanded_hostnames = expand_hostlist(slurm_nodelist) if slurm_nodelist != "non défini" else []

    # Définition des variables MASTER
    master_addr = expanded_hostnames[0] if expanded_hostnames else "non défini"

    # Affichage des informations
    print(f"SLURM_JOB_NODELIST: {slurm_nodelist}")
    print(f"SLURM_NODEID: {slurm_nodeid}")
    print(f"SLURM_JOB_NUM_NODES: {slurm_num_nodes}")
    print(f"SLURM_PROCID: {slurm_procid}")
    print(f"Expanded hostnames: {expanded_hostnames}")
    print(f"MASTER_ADDR: {master_addr}")
    print(f"MASTER_PORT: {slurm_port}")
