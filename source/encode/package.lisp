(cl:defpackage #:statistical-learning.encode
  (:use #:cl #:statistical-learning.aux-package)
  (:nicknames #:sl.encode)
  (:export
   #:encode
   #:make-encoders
   #:size-required
   #:encode-column-value
   #:size-required
   #:make-encoder-for-type))
