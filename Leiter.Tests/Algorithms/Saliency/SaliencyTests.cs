using Xunit;
using Leiter.Core;
using Leiter.Pixels;
using Leiter.Algorithms;
using Leiter.Algorithms.Segmentation;
using Leiter.Algorithms.Saliency;
using System;
using System.Linq;

namespace Leiter.Tests.Algorithms.Saliency;

/// <summary>
/// Provides unit tests for <see cref="SaliencyTests" />.
/// </summary>
public class SaliencyTests
{
    /// <summary>
    /// Verifies that the global contrast saliency should compute correctly.
    /// </summary>
    [Fact]
    public void GlobalContrastSaliency_ShouldComputeCorrectly()
    {
        // 20x20 image with 25+ unique quantized colors to satisfy the smoothing logic
        var img = new SequentialMatrix<Rgb8>(20, 20);
        for (int y = 0; y < 20; y++)
        {
            for (int x = 0; x < 20; x++)
            {
                // Each grid cell has a different color to ensure we have many unique colors
                byte r = (byte)(x * 12);
                byte g = (byte)(y * 12);
                byte b = (byte)((x + y) * 6);
                img[x, y] = new Rgb8(r, g, b);
            }
        }

        // Compute saliency (both with and without color smoothing)
        var saliencyWithSmoothing = GlobalContrastSaliency.ComputeSaliency(img, enableColorSpaceSmoothing: true);
        var saliencyWithoutSmoothing = GlobalContrastSaliency.ComputeSaliency(img, enableColorSpaceSmoothing: false);

        Assert.Equal(20, saliencyWithSmoothing.Width);
        Assert.Equal(20, saliencyWithSmoothing.Height);

        // Min and max should be normalized to 0.0 and 1.0
        var minS = saliencyWithSmoothing.Min(p => p.Value);
        var maxS = saliencyWithSmoothing.Max(p => p.Value);
        Assert.Equal(0.0, minS, 4);
        Assert.Equal(1.0, maxS, 4);

        // Verify that different areas have different saliency (i.e. we are not getting flat zeros)
        Assert.True(saliencyWithSmoothing.Any(p => p.Value > 0.0 && p.Value < 1.0));
        Assert.True(saliencyWithoutSmoothing.Any(p => p.Value > 0.0 && p.Value < 1.0));
    }

    /// <summary>
    /// Verifies that global contrast saliency of a contrasting object is correctly computed.
    /// </summary>
    [Fact]
    public void GlobalContrastSaliency_ContrastingObject_ShouldBeMostSalient()
    {
        // 10x10 image with a 4x4 red square in the center (16 pixels, 16%) and black background
        var img = new SequentialMatrix<Rgb8>(10, 10);
        img.SetAll(new Rgb8(0, 0, 0));
        for (int y = 3; y <= 6; y++)
        {
            for (int x = 3; x <= 6; x++)
            {
                img[x, y] = new Rgb8(255, 0, 0);
            }
        }

        var saliency = GlobalContrastSaliency.ComputeSaliency(img, enableColorSpaceSmoothing: false);

        // The contrasting red region should have highest saliency (1.0)
        Assert.Equal(1.0, saliency[5, 5].Value, 4);

        // The black background should have the lowest saliency (0.0)
        Assert.Equal(0.0, saliency[0, 0].Value, 4);
    }

    /// <summary>
    /// Verifies that the regional contrast saliency should compute correctly.
    /// </summary>
    [Fact]
    public void RegionalContrastSaliency_ShouldComputeCorrectly()
    {
        // 20x20 image with 25+ unique quantized colors to satisfy the smoothing logic
        var img = new SequentialMatrix<Rgb8>(20, 20);
        for (int y = 0; y < 20; y++)
        {
            for (int x = 0; x < 20; x++)
            {
                byte r = (byte)(x * 12);
                byte g = (byte)(y * 12);
                byte b = (byte)((x + y) * 6);
                img[x, y] = new Rgb8(r, g, b);
            }
        }

        // Get segmentation
        var segmentation = EgbiSegmentation.Segment(img, kFactor: 1.0, epsilon: 1.0, minSegmentSize: 5);

        // Compute regional saliency
        var saliency = RegionalContrastSaliency.ComputeSaliency(img, segmentation, enableSmoothing: true, computeBorderRegions: true);
        
        Assert.Equal(20, saliency.Width);
        Assert.Equal(20, saliency.Height);

        // Ensure normalized
        var minS = saliency.Min(p => p.Value);
        var maxS = saliency.Max(p => p.Value);
        Assert.Equal(0.0, minS, 4);
        Assert.Equal(1.0, maxS, 4);

        // Verify regional saliency contains various levels (i.e. not all flat zeros/ones)
        Assert.True(saliency.Any(p => p.Value > 0.0 && p.Value < 1.0));

        // Turn off smoothing and border calculation to test the alternate branches
        var saliencyRaw = RegionalContrastSaliency.ComputeSaliency(img, segmentation, enableSmoothing: false, computeBorderRegions: false);
        Assert.Equal(20, saliencyRaw.Width);
        Assert.True(saliencyRaw.Min(p => p.Value) >= 0.0);
    }

    /// <summary>
    /// Verifies that regional contrast saliency of a contrasting object is correctly computed.
    /// </summary>
    [Fact]
    public void RegionalContrastSaliency_ContrastingObject_ShouldBeMostSalient()
    {
        // 12x12 image with a 4x4 red square in the center (16 pixels) and black background
        var img = new SequentialMatrix<Rgb8>(12, 12);
        img.SetAll(new Rgb8(0, 0, 0));
        for (int y = 4; y <= 7; y++)
        {
            for (int x = 4; x <= 7; x++)
            {
                img[x, y] = new Rgb8(255, 0, 0);
            }
        }

        // Segment the image. With exact colors, the segment boundaries are completely clear.
        var segmentation = EgbiSegmentation.Segment(img, kFactor: 1.0, epsilon: 1.0, minSegmentSize: 4);

        // Compute regional saliency
        var saliency = RegionalContrastSaliency.ComputeSaliency(img, segmentation, enableSmoothing: false, computeBorderRegions: false);

        // The contrasting center red region should have highest saliency (1.0)
        Assert.Equal(1.0, saliency[5, 5].Value, 4);

        // The background black region should have the lowest saliency (0.0)
        Assert.Equal(0.0, saliency[0, 0].Value, 4);
    }
}
