;(function (window, document) {
  "use strict";

  /*
   * FaceDetect
   * A class for detecting faces
   * @param DOMObject webcam
   * @param DOMObject canvas
   */
  var FaceDetect = function(webcam, canvas) {
    var self = this;

    if (!self.hasGetUserMedia()) {
      throw("getUserMedia() is not supported in your browser");
    }

    self.webcam = webcam;
    self.canvas = canvas;

    //Init fallback functions
    navigator.FaceDetectUserMedia = (navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia || navigator.msGetUserMedia);
    window.requestAnimFrame = (function(){
      return  window.requestAnimationFrame       ||
              window.webkitRequestAnimationFrame ||
              window.mozRequestAnimationFrame    ||
              function( callback ){
                window.setTimeout(callback, 1000 / 60);
              };
    })();

    self.initCamera();
  }

  FaceDetect.prototype.hasGetUserMedia = function() {
    var self = this;
    return !!(navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia || navigator.msGetUserMedia);
  };

  FaceDetect.prototype.initCamera = function() {
    var self = this;

    var attempts = 0;
    var findVideoSize = function() {
      if(self.webcam.videoWidth > 0 && self.webcam.videoHeight > 0) {
        self.cameraHeight = self.webcam.videoHeight;
        self.cameraWidth = self.webcam.videoWidth;
        self.ready();
      } else {
          if(attempts < 10) {
            attempts++;
            window.setTimeout(findVideoSize.bind(self), 200);
          } else {
            self.cameraHeight = 480;
            self.cameraWidth = 640;
            self.ready();
          }
      }
    };

    navigator.FaceDetectUserMedia({video: true, audio: false}, function(localMediaStream) {
      self.webcam.src = window.URL.createObjectURL(localMediaStream);

      self.webcam.addEventListener("loadeddata", function() {
        console.log("Loaded video");
        findVideoSize();
      })
    }, function() {
      throw("Couldn't load stream")
    });
  };

  FaceDetect.prototype.ready = function() {
    var self = this;

    console.log(self.cameraHeight, self.cameraWidth);
  };


  var detection = new FaceDetect(document.getElementById("webcam"), document.getElementById("canvas"));

}(this, document));
