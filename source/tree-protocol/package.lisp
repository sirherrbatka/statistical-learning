(cl:in-package #:cl-user)


(defpackage #:cl-grf.tree-protocol
  (:use #:cl #:cl-grf.aux-package)
  (:export
   #:depth
   #:force-tree
   #:fundamental-leaf-node
   #:fundamental-node
   #:fundamental-training-parameters
   #:fundamental-training-state
   #:fundamental-tree-node
   #:split-with-mode
   #:needs-split-p-with-mode
   #:split-mode
   #:needs-split-p-mode
   #:force-tree*
   #:leafp
   #:make-node
   #:needs-split-p
   #:split
   #:training-data
   #:maximal-depth
   #:treep
   ))
