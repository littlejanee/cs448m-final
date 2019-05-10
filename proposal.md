# Human-Plotter Collaborative Drawing

## Team members and their responsibilities

Will: hardware/software interface<br/>
Jane: interaction design/implementation<br/>
Both: hardware (mostly pen holders - with assistance from Teguh as needed)

## Description of project and overall goal

We are interested in designing and building a collaborative drawing system where the plotter is able to respond to the human’s drawing input and add its own strokes to the collaborative piece. There has been some exploration of human-computer collaborative drawing on a screen [1], looking at different interaction paradigms where the system is sometimes leading the drawing, sometimes following the user, etc. We are interested in thinking about some of these questions and how the interaction changes once the system is a physical machine and not just software in a tablet. Jennifer Jacobs has also explored user-created dynamic brushes, where the user is able to define how a brush should respond to various input (drawing, x/y, time, pressure, etc). An artist named Sougwen Chung has explored this in a number of art pieces titled “Drawing Operations” [3] (she has versions from 2015, 2017, and two in 2018) where she collaboratively draws with robot arms. The initial piece has the robot mimic her drawing -- the robot is drawing alongside the artist, mirroring her input (captured via an overhead camera).

We want to explore the nature of the t-bot plotter as a drawing machine, along with a variety of pen holders and design ways for it to draw that play to its strengths rather than specifically focusing on perfectly imitating the human. For instance, we’ve already discovered that one of its weaknesses is that it has a hard time with exact relative positioning between strokes. However, with a robust pen holder, it’ll be much better than a human at drawing exact shapes, like a line or a circle. Rather than fight against some of the qualities of the plotter’s drawing, we want to explore these differences and allow them to influence our design of its drawing response styles. 

## Plan and milestones

May 9: proposal due<br/>
May 17: capture user input<br/>
May 24: have the plotter able to recreate user input<br/>
May 31: explore various pen holders/interactions<br/>
June 7: finalize plotter’s interaction (drawing response styles)
June 10: project due

## Parts and supplies needed

In order to capture user input, we are considering either using an overhead camera. This requires the camera as well as pieces to build a frame. We are also considering using a pen that records its location such as: https://www.livescribe.com/site/livescribe3/.

Other than for capture, since we are using the t-bot plotter as the foundation to our collaborative drawing system, the supplies we will need are primarily components to build pen holders. We are interested in trying out a number of different pen holder designs though to see how they impact the drawing output.

## References and attributions

[1] Oh, Changhoon, et al. "I lead, you help but only with enough details: Understanding user experience of co-creation with artificial intelligence." CHI 2018.<br/>
[2] Jacobs, Jennifer, et al. "Dynamic Brushes: Extending Manual Drawing Practices with Artist-Centric Programming Tools." CHI 2018.<br/>
[3] Chung, Sougwen. “Drawing Operations (2015).” https://sougwen.com/project/drawing-operations
