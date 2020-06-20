(cl:in-package #:sl.proxy-tree)


(defmethod forward-call (proxy function &rest arguments)
  (apply function (inner proxy) arguments))


(defun honest (parameters)
  (make 'honest-tree
        :inner parameters))
