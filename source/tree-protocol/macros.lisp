(cl:in-package #:statistical-learning.tree-protocol)


(defmacro training-state-clone (training-state &rest arguments)
  (once-only (training-state)
    `(progn
       (check-type ,training-state fundamental-training-state)
       (cl-ds.utils:quasi-clone* ,training-state
         ,@arguments))))
