;(function (window, document) {
  "use strict";

  var FaceDetect = function() {
    var self = this;

    if (!self.hasGetUserMedia()) {
      throw("getUserMedia() is not supported in your browser");
    }

    navigator.FaceDetectUserMedia = (navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia || navigator.msGetUserMedia);

    self.initCamera();
  }

  FaceDetect.prototype.hasGetUserMedia = function() {
    var self = this;
    return !!(navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia || navigator.msGetUserMedia);
  };

  FaceDetect.prototype.initCamera = function() {
    var self = this;

    self.video = document.querySelector('video');

    navigator.FaceDetectUserMedia({video: true, audio: false}, function(localMediaStream) {
      self.video.src = window.URL.createObjectURL(localMediaStream);

      self.video.onloadedmetadata = function(e) {
        console.log("Loaded stream");
      };
    }, function() {
      throw("Couldn't load stream")
    });
  };


  var detection = new FaceDetect();

}(this, document));
