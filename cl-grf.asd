(cl:in-package #:cl-user)


(asdf:defsystem cl-grf
  :name "cl-grf"
  :version "0.0.0"
  :license "BSD simplified"
  :author "Marek Kochanowicz"
  :depends-on ( :iterate       :serapeum
                :lparallel     :cl-data-structures
                :metabang-bind :alexandria
                :documentation-utils-extensions)
  :serial T
  :pathname "source"
  :components ((:file "aux-package")
               (:module "data"
                :components ((:file "package")
                             (:file "macros")
                             (:file "types")
                             (:file "functions")
                             (:file "extras")))
               (:module "random"
                :components ((:file "package")
                             (:file "discrete-distribution")))
               (:module "model-protocol"
                :components ((:file "package")
                             (:file "generics")
                             (:file "types")
                             (:file "functions")
                             (:file "methods")
                             (:file "documentation")))
               (:module "performance"
                :components ((:file "package")
                             (:file "generics")
                             (:file "types")
                             (:file "methods")
                             (:file "functions")
                             (:file "documentation")
                             ))
               (:module "tree-protocol"
                :components ((:file "package")
                             (:file "generics")
                             (:file "types")
                             (:file "functions")
                             (:file "methods")
                             ))
               (:module "algorithms"
                :components ((:file "package")
                             (:file "types")
                             (:file "utils")
                             (:file "generics")
                             (:file "methods")))
               (:module "forest"
                :components ((:file "package")
                             (:file "generics")
                             (:file "types")
                             (:file "utils")
                             (:file "functions")
                             (:file "methods")))))
