(cl:defpackage #:statistical-learning.common
  (:use #:cl #:statistical-learning.aux-package)
  (:nicknames #:sl.common)
  (:export
   #:next-proxy
   #:proxy
   #:defgeneric/proxy
   #:lifting-proxy
   #:lift
   #:strip
   #:proxy-enabled))
