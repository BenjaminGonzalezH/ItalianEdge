######### Libraries #########
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages

######### Functions #########

"""
This block contains all main functions.
"""

def install_packages(force = False):
    """
    install_packages (function): Install all necessary packages for using GoSemSim library for
    wang index calculation use.
    
    Parameters:
    - force: if it is True, force the installation of all packages even if they are installed.

    Returns:
    - None - just the installation status.
    """
    try:
        # List of required packages and final status report.
        require_pkgs = ["BiocManager", "GoSemSim", 
                        "GenomicFeatures", "GO.db", 
                        "AnnotationDbi"]
        status_report = {}

        for pkg in require_pkgs:
            #############################################  Force install or first install process.
            if force or not rpackages.isinstalled(pkg):
                if pkg == "BiocManager":
                    utils = rpackages.importr('utils')
                    utils.install_packages(pkg)
                else:
                    robjects.r(f'BiocManager::install("{pkg}")')
                    status_report[pkg] = "Install Complete/Re-installed"
            #############################################  No installation process needed.
            else:
                status_report[pkg] = "Previously Installed"

    except Exception as e:
        raise RuntimeError(f"Runtime Error: {e}")
    else:
        print(f"Packages install status: {status_report}\n")


