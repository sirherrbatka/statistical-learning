(cl:in-package #:sl.proxy-tree)


(defmethod forward-call (proxy function &rest arguments)
  (apply function (inner proxy) arguments))
