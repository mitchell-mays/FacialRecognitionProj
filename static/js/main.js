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

    self.maxWorkSize = 160;
    self.currRect = null;
    self.rectChangeThresh = 4;

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

    self.ctx = self.canvas.getContext('2d');

    self.ctx.fillStyle = "rgb(0,255,0)";
    self.ctx.strokeStyle = "rgb(0,255,0)";

    // Calculate scale for work canvas
    var scale = Math.min(self.maxWorkSize/self.cameraWidth, self.maxWorkSize/self.cameraHeight);
    var scaledWidth = (self.canvas.width * scale) | 0;
    var scaledHeight = (self.canvas.height * scale) | 0;

    self.img_u8 = new jsfeat.matrix_t(scaledWidth, scaledHeight, jsfeat.U8_t | jsfeat.C1_t);
    self.workCanvas = document.createElement('canvas');
    self.workCanvas.width = scaledWidth;
    self.workCanvas.height = scaledHeight;
    self.work_ctx = self.workCanvas.getContext('2d');

    jsfeat.bbf.prepare_cascade(jsfeat.bbf.face_cascade);

    //Start tick function
    window.requestAnimFrame(self.tick.bind(self));
  };

  FaceDetect.prototype.tick = function() {
    var self = this;

    window.requestAnimFrame(self.tick.bind(self));

    if (self.webcam.readyState === self.webcam.HAVE_ENOUGH_DATA) {
      var rects = [];
      var angles = [0, -0.5, 0.5];
      self.ctx.drawImage(self.webcam, 0, 0, self.canvas.width, self.canvas.height);
      //self.ctx.drawImage(self.webcam, 0, 0, self.workCanvas.width, self.workCanvas.height);

      var x = self.workCanvas.width / 2;
      var y = self.workCanvas.height / 2;
      var width = self.webcam.width;
      var height = self.webcam.height;

      for (var i = 0; i < angles.length; i++) {

        //self.work_ctx.drawImage(self.webcam, 0, 0, self.workCanvas.width, self.workCanvas.height);

        var angleInRadians = angles[i];
        self.work_ctx.translate(x, y);
        self.work_ctx.rotate(angleInRadians);
        self.work_ctx.drawImage(self.webcam, -self.workCanvas.width / 2, -self.workCanvas.height / 2, self.workCanvas.width, self.workCanvas.height);
        self.work_ctx.rotate(-angleInRadians)
        self.work_ctx.translate(-x, -y);
    
        var imageData = self.work_ctx.getImageData(0, 0, self.workCanvas.width, self.workCanvas.height);

        jsfeat.imgproc.grayscale(imageData.data, self.workCanvas.width, self.workCanvas.height, self.img_u8);

        var pyr = jsfeat.bbf.build_pyramid(self.img_u8, 24*2, 24*2, 4);

        var tempRects = jsfeat.bbf.detect(pyr, jsfeat.bbf.face_cascade);

        tempRects = jsfeat.bbf.group_rectangles(tempRects, 1);

        
        for (var currentRect = 0; currentRect < tempRects.length; currentRect++){
            tempRects[currentRect].x = tempRects[currentRect].x-x;
            tempRects[currentRect].y = tempRects[currentRect].y-y;
            //console.log(tempRects[currentRect].x);
            //console.log(tempRects[currentRect].y);

            var r = Math.sqrt(Math.pow(tempRects[currentRect].x,2) + Math.pow(tempRects[currentRect].y,2));
            var o = Math.atan(tempRects[currentRect].y/tempRects[currentRect].x);

            if (tempRects[currentRect].x < 0){
              o = o - Math.PI;
            }

            //console.log(tempRects[currentRect].x + r * Math.cos(angles[i]));
            //console.log(tempRects[currentRect].y + r * Math.sin(angles[i]));

            //console.log(tempRects[currentRect].x);
            //console.log(tempRects[currentRect].y);

            tempRects[currentRect].x = (r * Math.cos(o - angles[i]));
            tempRects[currentRect].y = (r * Math.sin(o - angles[i]));

            //console.log(tempRects[currentRect].x);
            //console.log(tempRects[currentRect].y);

            tempRects[currentRect]["angle"] = angles[i];
        }
        
        

        rects = rects.concat(tempRects);

      }

      // draw only most confident one
      self.drawFaces(rects, self.canvas.width / self.img_u8.cols, 1);

    }
  };

  FaceDetect.prototype.drawFaces = function(rects, sc, max) {
    var self = this;
    var on = rects.length;
    if(on && max) {
        jsfeat.math.qsort(rects, 0, on-1, function(a,b){return (b.confidence<a.confidence);})
    }
    var n = max || on;
    n = Math.min(n, on);
    var r;

    //var x = self.workCanvas.width / 2;
    //var y = self.workCanvas.height / 2;
    var x = self.canvas.width / 2;
    var y = self.canvas.height / 2;

    for(var i = 0; i < n; ++i) {
          r = rects[i];

          if (self.currRect == null){
            self.currRect = r;
          }
          var totDiff = Math.abs(self.currRect.x - r.x) + Math.abs(self.currRect.x - r.x) + Math.abs(self.currRect.width - r.width) + Math.abs(self.currRect.height - r.height);
          if (totDiff < self.rectChangeThresh){
            r = self.currRect;
          }
          else{
            self.currRect = r;
          }



          self.currRect = r;

          self.ctx.translate(x, y);
          //self.ctx.drawImage(self.webcam, -self.workCanvas.width / 2, -self.workCanvas.height / 2, self.workCanvas.width, self.workCanvas.height);
          self.ctx.rotate(-r.angle);
          //self.ctx.drawImage(self.webcam, -self.workCanvas.width / 2, -self.workCanvas.height / 2, self.workCanvas.width, self.workCanvas.height);
          var yChange = 0;
          var xChange = 0;
          if (r.angle < 0) {
            xChange = -1 * (r.width / 2);
          }
          else if (r.angle > 0){
            yChange = -1 * (r.height / 2);
            xChange = (r.width / 4);
          }

          //self.ctx.strokeRect((r.x + xChange) | 0, (r.y + yChange) | 0, (r.width) | 0, (r.height) | 0);

          //self.ctx.strokeRect(0, 0, self.workCanvas.width, self.workCanvas.height);
          self.ctx.strokeRect(((r.x + xChange) * sc) | 0, ((r.y + yChange) * sc) | 0, (r.width * sc) | 0, (r.height * sc) | 0);
          //console.log(r.x * sc);
          //console.log(r.y * sc);
          self.ctx.rotate(r.angle);
          self.ctx.translate(-x, -y);
        
    }


  };


  var detection = new FaceDetect(document.getElementById("webcam"), document.getElementById("canvas"));

}(this, document));
