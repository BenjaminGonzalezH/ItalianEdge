######################## Source file description.
# This code is completely dedicated to manage all necessary libraries 
# installation to use Wang Index (Go terms dedicated genes similarity metric) 
# and (if necessary) re-install for updated versions of this libraries.

######################## Main variables and structures.
PACKAGES_VECTOR = c("BioManager", "AnnotationDbi", 
                    "DBI", "digest", "GO.db", 
                    "methods", "rlang", "R.utils", 
                    "stats", "utils", "yulab.utils")

######################## Installation Process.
for (package in PACKAGES_VECTOR){
  print(package)
}