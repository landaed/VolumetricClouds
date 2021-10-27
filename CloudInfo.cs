using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CloudInfo : MonoBehaviour
{
    public Transform player;
    Texture2D[] textList;
    string[] files;
    string pathPreFix;
    int i = 0;
    // Start is called before the first frame update
    private void OnRenderImage(RenderTexture source, RenderTexture destination)
    {
      gameObject.GetComponent<Renderer>().material.SetVector("BoundsMin", transform.position - transform.localScale /2);
      gameObject.GetComponent<Renderer>().material.SetVector("BoundsMax", transform.position + transform.localScale /2);
    }
    private IEnumerator LoadImages(){
        //load all images in default folder as textures and apply dynamically to plane game objects.
        //6 pictures per page
        textList = new Texture2D[files.Length];

        int dummy = 0;
        foreach(string tstring in files){

                string pathTemp = pathPreFix + tstring;
            WWW www = new WWW(pathTemp);
                    yield return www;
                    Texture2D texTmp = new Texture2D(1024, 1024, TextureFormat.DXT1, false);
                    www.LoadImageIntoTexture(texTmp);

                    textList[dummy] = texTmp;

            dummy++;
        }

    }
    void Start(){
      Camera.main.depthTextureMode = DepthTextureMode.Depth;
      string path = @"D:\unityHub\simChopPublic\SimChop\Assets\SimChop\PhysicsGif";
      pathPreFix = @"file://";
      files = System.IO.Directory.GetFiles(path, "*.png");
      StartCoroutine(LoadImages());

    }
    // Update is called once per frame
    void FixedUpdate(){
      gameObject.GetComponent<Renderer>().material.SetTexture("_MainTex", textList[i]);
      i++;
      if(i >= textList.Length){
        i=0;
      }
    }
    void Update()
    {


      gameObject.GetComponent<Renderer>().material.SetVector("BoundsMin", transform.position - transform.localScale /2);
      gameObject.GetComponent<Renderer>().material.SetVector("BoundsMax", transform.position + transform.localScale /2);
      gameObject.GetComponent<Renderer>().material.SetVector("Player", player.position);
    }
}
