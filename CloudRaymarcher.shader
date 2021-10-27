Shader "Unlit/CloudRaymarcher"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
        _CloudScale ("Cloud Scale", Range(0.1, 100.0)) = 1
        _CloudOffset ("Cloud Offset", Range(0.1, 10000.0)) = 1
        _DensityThreshold ("Density Threshold", Range(0., 1.0)) = 1
        _DensityMultiplier ("Density Multiplier", Range(0., 1.0)) = 1
        _NumSteps ("NumSteps", Range(1, 1000)) = 1
        _Irregular ("Irregularity", Range(0.1, 1.0)) = 1
        _Smoothness ("Smoothness", Range(0.1, 1.0)) = 1
        _Speed ("Speed", Range(0,100)) = 0
        _DarknsessThreshold ("Darkness threshold", Range(0., 1.0)) = 1
        _LightAbsorbtion ("Light Absorbtion of Clouds", Range(0.1, 100.0)) = 1
        _LightAbsorbtionTowardsSun ("Light Absorbtion towards Sun", Range(0.1, 100.0)) = 1
        _NumStepsLight("NumSteps light", Range(0.1, 100.0)) = 1
        _PhaseVal("phaseVal", Range(0.1, 100.0)) = 1
        _DepthFalloff("_DepthFalloff", Range(0, 1)) = 1
        _ScewXY("Scew xy", Range(0.1, 100.0)) = 1
        _ScewXZ("Scew xz", Range(0.1, 100.0)) = 1
        _ScewYZ("Scew yz", Range(0.1, 100.0)) = 1
    }
    SubShader
    {
        Tags { "Queue"="Transparent" "RenderType"="Transparent" "LightMode"="ForwardBase"}
  		  Blend SrcAlpha OneMinusSrcAlpha
        LOD 100

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            // make fog work
            #pragma multi_compile_fog

            #include "UnityCG.cginc"
            #include "Lighting.cginc"
            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                UNITY_FOG_COORDS(1)
                float4 vertex : SV_POSITION;
                float3 ro : TEXCOORD1;
                float3 hitPos: TEXCOORD2;
                float4 ScreenPos: TEXCOORD3;

            };

            sampler2D _MainTex;
            float4 _MainTex_ST;
            float _CloudScale;
            float _CloudOffset;
            float _DensityThreshold;
            float _DensityMultiplier;
            float _NumSteps;
            float _Irregular;
            float _Smoothness;
            float _Speed;
            float _DarknsessThreshold;
            float _LightAbsorbtion;
            float _LightAbsorbtionTowardsSun;
            float _NumStepsLight;
            float _PhaseVal;
            float _DepthFalloff;
            float _ScewXY;
            float _ScewXZ;
            float _ScewYZ;
            uniform float3 BoundsMin;
            uniform float3 BoundsMax;
            uniform float3 Player;
            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = TRANSFORM_TEX(v.uv, _MainTex);
                o.ro = _WorldSpaceCameraPos;
                o.hitPos = mul(unity_ObjectToWorld,v.vertex);
                UNITY_TRANSFER_FOG(o,o.vertex);
                o.ScreenPos = ComputeScreenPos(o.vertex);
                return o;
            }
            float smin( float a, float b, float k )
      			{
      					float h = clamp( 0.5+0.5*(b-a)/k, 0.0, 1.0 );
      					return lerp( b, a, h ) - k*h*(1.0-h);
      			}
            ////////

          // The cellular tile routine. Draw a few gradient shapes (eight circles, in this case) using
          // the darken (min(src, dst)) blend at various locations on a tile. Make the tile wrappable by
          // ensuring the shapes wrap around the edges. That's it.
          //
          // Believe it or not, you can get away with as few as four circles. Of course, there is 4-tap
          // Voronoi, which has the benefit of scalability, and so forth, but if you sum the total
          // instruction count here, you'll see that it's lower overall. Not requiring a hash function
          // provides the biggest benefit, but there is also less setup.
          //
          // However, the main reason you'd bother in the first place is the ability to extrapolate
          // to a 3D setting (swap circles for spheres) for virtually no extra cost. The result isn't
          // perfect, but 3D cellular tiles can enable you to put a Voronoi looking surface layer on a
          // lot of 3D objects for little cost. In fact, it's fast enough to raymarch.
          //
          float drawShape(in float2 p){

              // Wrappable circle distance. The squared distance, to be more precise.
              p = frac(p) - .5;
              return dot(p, p);

              // Other distance metrics.

              //p = abs(fract(p) - .5);
              //p = pow(p, float2(8));
              //return pow(p.x+p.y, .125)*.25;

              //p = abs(fract(p) - .5);
              //p *= p;
              //return max(p.x, p.y);

              //p = fract(p) - .5;
              //float n = max(abs(p.x)*.866 + p.y*.5, -p.y);
              //return n*n;

          }
          float drawSphere(in float3 p){

              p = frac(p)-.5;
              return dot(p, p);

              //p = abs(fract(p)-.5);
              //return dot(p, float3(.166));

          }
          // Draw some cirlcles on a repeatable tile. The offsets were partly based on science, but
          // for the most part, you could choose any combinations you want.
          //
          float cellTex(in float3 p){

            float c = .25; // Set the maximum.
            // Draw four overlapping objects (spheres, in this case) using the darken blend
            // at various positions throughout the tile.
            c = min(c, drawSphere(p - float3(.81, .62, .53)));
            c = min(c, drawSphere(p - float3(.39, .2, .11)));

            c = min(c, drawSphere(p - float3(.62, .24, .06)));
            c = min(c, drawSphere(p - float3(.2, .82, .64)));


            // Add some smaller spheres at various positions throughout the tile.

            p *= 1.4142;

            c = min(c, drawSphere(p - float3(.48, .29, .2)));
            c = min(c, drawSphere(p - float3(.06, .87, .78)));

            // More is better, but I'm cutting down to save cycles.
            c = min(c, drawSphere(p - float3(.6, .86, .0)));
            c = min(c, drawSphere(p - float3(.18, .44, .58)));

            // More is better, but I'm cutting down to save cycles.
            c = min(c, drawSphere(p - float3(.26, .46, .2)));
            c = min(c, drawSphere(p - float3(.48, .74, .0)));

            // More is better, but I'm cutting down to save cycles.
            c = min(c, drawSphere(p - float3(.62, .26, .32)));
            c = min(c, drawSphere(p - float3(.98, .14, .18)));

            // More is better, but I'm cutting down to save cycles.
            c = min(c, drawSphere(p - float3(.16, .16, .12)));
            c = min(c, drawSphere(p - float3(.18, .14, .10)));

            // More is better, but I'm cutting down to save cycles.
            c = min(c, drawSphere(p - float3(.22, .26, .22)));
            c = min(c, drawSphere(p - float3(.28, .24, .28)));

            return (c*4.); // Normalize.

          }
            float ndot(float2 a, float2 b ) { return a.x*b.x - a.y*b.y; }

            float2 rand2(float2 p)
            {
            	float2 q = float2(dot(p,float2(127.1,311.7)),
            		dot(p,float2(269.5,183.3)));
            	return frac(sin(q)*43758.5453);
            }

            float rand(float2 p)
            {
            	return frac(sin(dot(p,float2(419.2,371.9))) * 833458.57832);
            }

            float iqnoise(in float2 pos, float irregular, float smoothness)
            {
            	float2 cell = floor(pos);
            	float2 cellOffset = frac(pos);

            	float sharpness = 1.0 + 63.0 * pow(1.0-smoothness, 4.0);

            	float value = 0.0;
            	float accum = 0.0;
            	// Sample the surrounding cells, from -2 to +2
            	// This is necessary for the smoothing as well as the irregular grid.
            	for(int x=-2; x<=2; x++ )
            	for(int y=-2; y<=2; y++ )
            	{
            		float2 samplePos = float2(float(y), float(x));

              		// Center of the cell is not at the center of the block for irregular noise.
              		// Note that all the coordinates are in "block"-space, 0 is the current block, 1 is one block further, etc
            		float2 center = rand2(cell + samplePos) * irregular;
            		float centerDistance = length(samplePos - cellOffset + center);

            		// High sharpness = Only extreme values = Hard borders = 64
            		// Low sharpness = No extreme values = Soft borders = 1
            		float sam = pow(1.0 - smoothstep(0.0, 1.414, centerDistance), sharpness);

            		// A different "color" (shade of gray) for each cell
            		float color = rand(cell + samplePos);
            		value += color * sam;
            		accum += sam;
            	}

            	return value/accum;
            }
            //3d noise
            float hash(float n)
            {
            	return frac(sin(n) * 43728.1453);
            }

            float noise(float3 x)
            {
            	float3 p = floor(x);
            	float3 f = frac(x);

            	f = f * f * (3.0 - 2.0 * f);
            	float n = p.x + p.y * 55.0 + p.z * 101.0 ;

              return lerp(
              	lerp(
              		lerp(hash(n), hash(n + 1.0), f.x),
              		lerp(hash(n+55.0), hash(n + 56.0), f.x),
              		f.y),
              	lerp(
              		lerp(hash(n+101.0), hash(n + 102.0), f.x),
              		lerp(hash(n+156.0), hash(n + 157.0), f.x),
              		f.y),
              	f.z);
            }
            float2 rayBoxDst(float3 boundsMin, float3 boundsMax, float3 ro, float3 rd){
              float3 t0 = (boundsMin-ro)/rd;
              float3 t1 = (boundsMax-ro)/rd;
              float3 tmin = min(t0,t1);
              float3 tmax = max(t0,t1);
              float dstA = max(max(tmin.x,tmin.y),tmin.z);
              float dstB = min(tmax.x,min(tmax.y,tmax.z));

              float dstToBox = max(0,dstA);
              float dstInsideBox = max(0,dstB-dstToBox);
              return float2(dstToBox,dstInsideBox);
            }
            float sampleDensity(float3 pos){
              //float off= _CloudOffset+_Time.y*_Speed;
              float3 uvw = pos * _CloudScale *0.001 + (_CloudOffset) * 0.01;
              uvw.xz +=_Time.y*_Speed;
            //  float j = 10;//tex2D(_MainTex,uvw.xy).r;
              //smoothstep(.8,.1,cellTex(uv*4.))
            //  float4 shape =(1-smoothstep(.8,.1,cellTex(uvw.xy*_ScewXY)))*(1-smoothstep(.8,.1,cellTex(uvw.xz*_ScewXZ)))*(1-smoothstep(.8,.1,cellTex(uvw.zy*_ScewYZ)));//  (noise(uvw)/10)+iqnoise(uvw.xy, _Irregular, _Smoothness)*iqnoise(uvw.xz, _Irregular, _Smoothness);
              float4 shape = smoothstep(.1,.9,cellTex(uvw))*smoothstep(.5,.3,cellTex(uvw*_ScewXY))*smoothstep(.5,.3,cellTex(uvw*_ScewXZ));
              float density = max(0,shape.r-_DensityThreshold)*_DensityMultiplier;

              return density;
            }
            sampler2D _CameraDepthTexture;

            float GetLight(float3 p){
            //position of the light source
            float3 l = _WorldSpaceLightPos0.xyz;

            //normal of object
            /*float x = iqnoise(p.xy, .9, .9);
            float y = iqnoise(p.xz, .9, .9);
            float z = iqnoise(p.zy, .9, .9);*/
            float3 n = float3(0,1,0);

            // dot product of the light floattor and normal of the point
            // will give us the amount of lighting to apply to the point
            // dot() evaluates to values between -1 and 1, so we will clamp it
            float diff = clamp(dot(n, l),0.,1.);

            // calculate if point should be a shadow:
            // raymarch from point being calculated towards light source
            // if hits surface of something else before the light,
            // then it must be obstructed and thus is a shadow
            // the slight offset "p+n*SURFACE_DIST*1.1" is needed to ensure the
            // break condistions in the function are not met too early
            /*float d = RayMarch(p+n*01*1.1,l);
            if(d < l){
                diff *= 0.1;
            }*/
            return diff;
        }
            float lightmarch(float3 pos){
              float3 dirToLight = _WorldSpaceLightPos0.xyz;
              float dstInsideBox = rayBoxDst(BoundsMin, BoundsMax,pos,1/dirToLight).y;

              float stepSize = dstInsideBox/_NumStepsLight;
              float totalDensity = 0;
              for(int i = 0; i < _NumStepsLight; i++){

                pos += dirToLight *stepSize;
                totalDensity += max(0,sampleDensity(pos)*stepSize);
              }
              float transmittance = exp(-totalDensity * _LightAbsorbtionTowardsSun);
              return _DarknsessThreshold + transmittance * (1-_DarknsessThreshold);
            }
            float2x2 Rot(float a) {
                float s=sin(a), c=cos(a);
                return float2x2(c, -s, s, c);
            }
            fixed4 frag (v2f i) : SV_Target
            {
                //ray origin (cameraPos)
                float3 ro = i.ro;
                //ray Direction (cameraLookVector)
                float3 rd = normalize(i.hitPos-ro);
                //ScreenUV
                float2 screen_pos = i.ScreenPos.xy/i.ScreenPos.w;

        				// get the depth texture
        				float nonLinearDepth = tex2D(_CameraDepthTexture, screen_pos).r;


              //  depth = pow(Linear01Depth(depth), .05);

                float depth =Linear01Depth(nonLinearDepth)*_ProjectionParams.z;

                //get the bounds of the box we raycast into
                float2 rayBoxInfo = rayBoxDst(BoundsMin,BoundsMax,ro,rd);

                //stores how far we marched
                float dstTravelled = 0;

                //how far box is from camera
                float distToBox=rayBoxInfo.x;
                float distInsideBox=rayBoxInfo.y;
                //how far we march each time
                float stepSize = distInsideBox/_NumSteps;

                //density of volume
                float totalDensity =0;
                //make sure we dont march outside bounds
                float dstLimit = min(depth-distToBox,distInsideBox);
                //light energy (used to calculate brightness)
                float3 lightE = 0;
                float transmittance = 1;
                float d01 =Linear01Depth(nonLinearDepth);
                float eyeD =LinearEyeDepth(nonLinearDepth);
                for(int i = 0; i <_NumSteps; i++){

                //while(dstTravelled < dstLimit){
                  //current position in the marching
                  float3 rayPos = ro + rd * (distToBox+dstTravelled);
                  float3 dPos = ro  + rd * d01*_ProjectionParams.z;
                  //sample density at the rayPos and add it to the total
                  float3 rP = rayPos;
                  float3 pP = Player;
                  pP.xz = mul(Rot(45),pP.xz);
                  rP.xz = mul(Rot(45),rP.xz);
                  rP.x *=_Speed;
                  pP.x *=_Speed;
                  //rP.z *=_Speed;
                  float d = length(rP-(pP.xyz-float3(0,0.,_Speed/2)));

                  //float distToP = length(ray)
                  //float falloff = smoothstep(1,3,abs(rayPos.z-Player.z)+abs(rayPos.x-Player.x));
                  float density = sampleDensity(rayPos)*smoothstep (5,9,length(Player-rayPos));

                  if(density>0){
                    float lightTransmittance = lightmarch(rayPos);
                    lightE+=density*stepSize*transmittance*lightTransmittance*_PhaseVal;
                    //further into steps and more light absorbtion we have, the smaller transmittance will be (see graph of e^-x)
                    transmittance*= exp(-density*stepSize*_LightAbsorbtion);
                    if(transmittance < 0.01){
                      break;
                    }
                  }
                //  totalDensity +=sampleDensity(rayPos)*stepSize;

                //  totalDensity*=smoothstep(0.5,.1,length(rayPos01-pPos));
                  //totalDensity*=
                  //get brightness at point

                  //lightE =GetLight(rayPos)*(1/(dstTravelled*.001));
                  dstTravelled+=stepSize;
                }
                //transmittance is e^-totalDensity
                //transmittance = exp(-totalDensity);
                float3 cloudCol = lightE * _LightColor0.xyz;


                // sample the texture

              //  float d = depth;

                //

                //fixed4 col =fixed4(lerp(lightE,lightE*float3(1,0,0),_WorldSpaceLightPos0.x),transmittance*alpha);
              //  if(dstTravelled > distToBox+nonLinearDepth)
                //  discard;

                float dT = (dstTravelled+distToBox)/_ProjectionParams.z;
               float3 rayPos = ro + rd * ((distToBox+dstTravelled));
              // depth = LinearEyeDepth(nonLinearDepth);
        			// float3 dPos = rd * depth + ro;
               //float alpha = smoothstep (0,2,length(rayPos-Player));

            //    dT*=alpha;
              //  float alpha =smoothstep(.9,.1,abs(((transmittance)*dT)-d01));
                //fixed4 col =fixed4(lerp(lightE,lightE*float3(1,0,0),_WorldSpaceLightPos0.x),1-transmittance);
                //eyeD=(eyeD/_ProjectionParams.z);
              //  d01=1-d01;
                fixed4 col = fixed4(cloudCol,1-transmittance);
                //fixed4 col = fixed4(dPos,1);
                // apply fog
              //  UNITY_APPLY_FOG(i.fogCoord, col);
              //  float dt = Linear01Depth(nonLinearDepth);
                return col;// fixed4(dt,dt,dt,1);
            }
            ENDCG
        }
    }
}
