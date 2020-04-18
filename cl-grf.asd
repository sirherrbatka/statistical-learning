(cl:in-package #:cl-user)


(asdf:defsystem cl-grf
  :name "cl-grf"
  :version "0.0.0"
  :license "BSD simplified"
  :author "Marek Kochanowicz"
  :depends-on ( :iterate       :serapeum
                :lparallel     :cl-data-structures
                :metabang-bind :alexandria
                :local-time    :cl-postgres
                :s-sql         :mcclim
                :documentation-utils-extensions
                :postmodern)
  :serial T
  :pathname "source"
  :components ((:file "aux-package")
               (:module "tree-protocol"
                :components ((:file "package")
                             (:file "generics")
                             (:file "types")
                             (:file "methods")
                             (:file "functions")))
               ))
